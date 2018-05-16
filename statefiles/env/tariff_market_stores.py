import logging
from datetime import datetime, timedelta
from functools import reduce
from typing import List
import logging

from ast import literal_eval

import model.tariff as t
import model.tariff_status as ts
import model.customer_info as ci
import model.tariff_transaction as tt
from statefiles.env.base_store import BaseStore
from util.config import DATETIME_PATTERN
from model.tariff_transaction import TransactionType
from model.rate import Rate
from model.StatelineParser import StatelineParser

log = logging.getLogger(__name__)

_rates_left_over = []  # sometimes a rate is applied to multiple Tariffs in the state files. That causes confusion


# furthermore, the rate.-rr is called before the tariffSpec.-rr so I can't add the rate to the tariff
# if the tariff doesn't exist yet

class TariffMarketStores():
    def __init__(self, env):
        self.env = env
        self.tariffTransactions = []
        self.rates = {}
        self.tariffs = {}
        self.tariff_stats = {}
        self.customers = {} 
        self.transactions = {}  # map of lists . Key => customerId, values the transaction objects

    def get_rate_for_customer_transactions(self, transactions: List[tt.TariffTransaction]) -> Rate:
        potential_rates = [self.tariffs[t.tariffSpec]._rates for t in
                           transactions]  # tariff and TariffSpec have same id (luckily)
        potential_rates = reduce(lambda sum, list: sum + list, potential_rates)
        time_of_transaction = self.env.timeslot_store.get_datetime_for_timeslot(transactions[0].when)
        potential_rates = [r for r in potential_rates if r.is_applicable(time_of_transaction)]

        # no rates found... odd. manually searching and adding again
        if len(potential_rates) is 0:
            for t in transactions:
                potential_rates = [r for r in self.rates.values() if r.tariffId == t.tariffSpec]
                map(lambda r: self.tariffs[self.rates[0].tariffId].add_rate(r), potential_rates)

        if len(potential_rates) is 0:
            logging.warning("we're missing rates over here")
            return None
        # tier threshold allows multiple rates. gotta think how to handle this
        # if len(potential_rates)> 1:
        #    print("too many rates?")

        #    for r in potential_rates:
        #        print(r)
        #    print("for transactions")
        #    for t in transactions:
        #        print(t)
        return potential_rates[0]

    def handle_tariff_rr(self, line: str):
        tariff = t.Tariff.from_state_line(line)
        self.tariffs[tariff.id_] = tariff
        for r in _rates_left_over:
            if r.tariffId == tariff.id_:
                tariff.add_rate(r)

    def handle_tariff_new(self, line: str):
        parts = StatelineParser.split_line(line)
        tariff = t.Tariff(id_=parts[1], brokerId=parts[3], powerType=parts[4])
        self.tariffs[tariff.id_] = tariff
        self._add_tariff_stats_for_tariff_id(tariff.id_)

    def handle_tariff_addRate(self, line: str):
        parts = StatelineParser.split_line(line)
        tariff = self.tariffs[parts[1]]
        tariff.add_rate(self.rates[3])

    def handle_tariffStatus_new(self, line: str):
        _ts = ts.TariffStatus.from_state_line(line)
        if _ts.status == ts.Status.success:
            self._add_tariff_stats_for_tariff_id(_ts.tariff_id)
            self.tariffs[_ts.tariff_id].status = t.Status.ACTIVE

    def _add_tariff_stats_for_tariff_id(self, id_):
        self.tariff_stats[id_] = t.TariffStats(self.env.current_timestep)

    def handle_tariffRevoke_new(self, line: str):
        parts = StatelineParser.split_line(line)
        self.tariffs[parts[4]].status = t.Status.WITHDRAWN

    # handling rate messages
    #########################################################################################################################

    def handle_rate_new(self, line: str):
        parts = StatelineParser.split_line(line)
        id_ = parts[1]
        self.rates[id_] = Rate(id_)

    def handle_rate_withValue(self, line: str):
        parts = StatelineParser.split_line(line)
        rate = self.rates[parts[1]]
        if rate is not None:
            rate.minValue = float(parts[3])

    def handle_rate_setTariffId(self, line: str):
        parts = StatelineParser.split_line(line)
        rate = self.rates[parts[1]]
        if rate is not None:
            rate.tariffId = parts[3]
            self.tariffs[rate.tariffId].add_rate(rate)

    def handle_rate_rr(self, line: str):
        rate = Rate.from_state_line(line)
        self.rates[rate.id_] = rate
        if rate.tariffId in self.tariffs:
            self.tariffs[rate.tariffId].add_rate(rate)
        else:
            _rates_left_over.append(rate)

    # handling CustomerInfo
    #########################################################################################################################

    def handle_customerInfo(self, line: str):
        """
        handles all customerInfo methods. If it's a "new" method, we create the object
        and add it to the repo of customers. Otherwise the line gives the method and the params
        that are needed and we just call that. all methods are prefixed with _jsm_ which stands for
        "java state method". this is internal only and meant for learning off the state messages.

        """
        parts = StatelineParser.split_line(line)
        method = parts[2]
        method_call = "_jsm_" + parts[2]
        if method == "new":
            info = ci.CustomerInfo(id_=parts[1], name=parts[3], population=int(parts[4]))
            self.customers[parts[1]] = info
        else:
            target = self.customers[parts[1]]
            to_call = getattr(target, method_call)
            if callable(to_call):
                to_call(*parts[3:])

    # handling TariffTransaction
    #########################################################################################################################
    def handle_TariffTransaction_new(self, line: str):

        # "8828:org.powertac.common.TariffTransaction::3328::new::1876          ::360       ::SIGNUP        ::1878                      ::3320                  ::30000             ::0.0           ::-0.0          ::false"
        # 100100:org.powertac.common.TariffTransaction::5486::new::1876         ::360       ::CONSUME       ::1878                      ::3709                  ::1                 ::-46.5874647092::23.2937423546::false
        # 100300:org.powertac.common.TariffTransaction::5492::new::1876         ::360       ::CONSUME       ::1878                      ::3728                  ::1                 ::-53.3333333333::26.66666666664::false
        # 100331:org.powertac.common.TariffTransaction::5498::new::1876         ::360       ::CONSUME       ::1878                      ::3742                  ::1                 ::-35.555555556 ::17.77777777778::false
        #                                                         Broker broker :: int when ::Type txType   ::TariffSpecification spec  :: CustomerInfo customer:: int customerCount:: double kWh   :: double charge:: boolean regulation)

        parts = StatelineParser.split_line(line)
        type_ = self._get_tariff_type(parts)
        params = [parts[1]]  # id of object
        params.extend(parts[4:])  # params of original constructor
        transaction = tt.TariffTransaction(*params)
        self._add_transaction(transaction)

        if type_ is TransactionType.PUBLISH:
            # handle type. Technically here we need to set tariff to published
            pass
        elif type_ is TransactionType.PRODUCE or TransactionType.CONSUME or TransactionType.SIGNUP or TransactionType.WITHDRAW or TransactionType.REFUND:
            self._handle_transaction_stats(transaction)
        elif type_ is TransactionType.PERIODIC:
            # handle type
            pass
        elif type_ is TransactionType.SIGNUP:
            # handle type
            pass
        elif type_ is TransactionType.WITHDRAW:
            # handle type
            pass
        elif type_ is TransactionType.REVOKE:
            # handle type
            pass
        elif type_ is TransactionType.REFUND:
            # handle type
            pass

    def _add_transaction(self, trans: tt.TariffTransaction):
        """gets always called (we store everything)"""
        customer_id = trans.customerInfo
        # initializing on first transaction for customer
        if customer_id not in self.transactions:
            self.transactions[customer_id] = []
        cust_trans = self.transactions[customer_id]
        # adding new list for current timestep
        if len(cust_trans) < self.env.current_timestep or not cust_trans:
            cust_trans.append([])

        trans_now = cust_trans[-1]
        trans_now.append(trans)

    def _handle_transaction_stats(self, trans: tt.TariffTransaction):
        """calculates specific statistics about the tariff in question"""
        tariff_stat = self.tariff_stats[trans.tariffSpec]
        tariff_stat.apply_tariff_transaction(trans)

    def _get_tariff_type(self, parts: List[str]) -> TransactionType:
        return TransactionType[parts[5]]

class TariffTransactionStore(BaseStore):
    def insert(self, obj):
        pass

    def __init__(self, ):
        super().__init__('id')


