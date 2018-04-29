"""
right now it just returns the list of line parsers. But it could be expanded to filter for specific types of messages so
that single components can be trained on the state lines
"""
def get_line_parser(env):
    return {
        "org.powertac.common.BalancingTransaction": None,
        "org.powertac.common.BankTransaction": None,
        "org.powertac.common.Broker": None,
        "org.powertac.common.CapacityTransaction": None,
        "org.powertac.common.CashPosition": None,
        "org.powertac.common.ClearedTrade": None,
        "org.powertac.common.Competition": {
            "withBootstrapTimeslotCount": env.handle_competition_withBootstrapTimeslotCount,
            "withBootstrapDiscardedTimeslots": env.handle_competition_withBootstrapDiscardedTimeslots,
            "withSimulationBaseTime": env.handle_competition_withSimulationBaseTime
        },
        "org.powertac.common.CustomerInfo": env.tariff_store.handle_customerInfo,
        "org.powertac.common.DistributionTransaction": None,
        "org.powertac.common.MarketPosition": None,
        "org.powertac.common.MarketTransaction": None,
        "org.powertac.common.msg.BalanceReport": None,
        "org.powertac.common.msg.BalancingControlEvent": None,
        "org.powertac.common.msg.BalancingOrder": None,
        "org.powertac.common.msg.DistributionReport": None,
        "org.powertac.common.msg.EconomicControlEvent": None,
        "org.powertac.common.msg.OrderStatus": None,
        "org.powertac.common.msg.SimEnd": None,  # not needed
        "org.powertac.common.msg.SimPause": None,  # not needed
        "org.powertac.common.msg.SimResume": None,  # not needed
        "org.powertac.common.msg.SimStart": None,
        "org.powertac.common.msg.TariffRevoke": {
            "new": env.tariff_store.handle_tariffRevoke_new
        },
        "org.powertac.common.msg.TariffStatus": {
            "new": env.tariff_store.handle_tariffStatus_new
        },
        "org.powertac.common.msg.TimeslotUpdate": {
            "new": env.timeslot_store.handle_timeslotUpdate_new
        },
        "org.powertac.common.Order": None,
        "org.powertac.common.Orderbook": None,
        "org.powertac.common.OrderbookOrder": None,
        "org.powertac.common.RandomSeed": None,  # not needed
        "org.powertac.common.Rate": {
            "-rr": env.tariff_store.handle_rate_rr,
            "new": env.tariff_store.handle_rate_new,
            "withValue": env.tariff_store.handle_rate_withValue,
            "setTariffId": env.tariff_store.handle_rate_setTariffId
        },
        "org.powertac.common.RegulationCapacity": None,
        "org.powertac.common.RegulationRate": None,
        "org.powertac.common.Tariff": None,
        "org.powertac.common.TariffSpecification": {
            "new": env.tariff_store.handle_tariff_new,
            "-rr": env.tariff_store.handle_tariff_rr
        },
        "org.powertac.common.TariffSubscription": None,
        "org.powertac.common.TariffTransaction": {
            "new": env.tariff_store.handle_TariffTransaction_new
        },
        "org.powertac.common.TimeService": None,
        "org.powertac.common.WeatherForecast": {
            "new": env.weather_store.handle_weatherForecast_new
        },  # not needed
        "org.powertac.common.WeatherForecastPrediction": {
            "new": env.weather_store.handle_weatherForecastPrediction_new
        },
        "org.powertac.common.WeatherReport": {
            "new": env.weather_store.handle_weatherReport_new
        },
        "org.powertac.du.DefaultBroker": None,
        "org.powertac.du.DefaultBrokerService": None,
        "org.powertac.evcustomer.customers.EvCustomer": None,
        "org.powertac.genco.Buyer": None,
        "org.powertac.genco.CpGenco": None,
        "org.powertac.genco.Genco": None,
        "org.powertac.genco.MisoBuyer": None,
    }

def get_wholesale_line_parser(env):
    return {
        "org.powertac.common.BalancingTransaction": None,
        "org.powertac.common.BankTransaction": None,
        "org.powertac.common.Broker": None,
        "org.powertac.common.CashPosition": None,
        "org.powertac.common.ClearedTrade": None,
        "org.powertac.common.Competition": {
            "withBootstrapTimeslotCount": env.handle_competition_withBootstrapTimeslotCount,
            "withBootstrapDiscardedTimeslots": env.handle_competition_withBootstrapDiscardedTimeslots,
            "withSimulationBaseTime": env.handle_competition_withSimulationBaseTime
        },
        "org.powertac.common.DistributionTransaction": None,
        "org.powertac.common.MarketPosition": None,
        "org.powertac.common.MarketTransaction": None,
        "org.powertac.common.msg.BalanceReport": None,
        "org.powertac.common.msg.BalancingControlEvent": None,
        "org.powertac.common.msg.BalancingOrder": None,
        "org.powertac.common.msg.DistributionReport": None,
        "org.powertac.common.msg.EconomicControlEvent": None,
        "org.powertac.common.msg.OrderStatus": None,
        "org.powertac.common.msg.SimStart": None,
        "org.powertac.common.msg.TimeslotUpdate": {
            "new": env.timeslot_store.handle_timeslotUpdate_new
        },
        "org.powertac.common.Order": None,
        "org.powertac.common.Orderbook": None,
        "org.powertac.common.OrderbookOrder": None,
        "org.powertac.common.RandomSeed": None,  # not needed
        "org.powertac.common.TimeService": None,
        "org.powertac.common.WeatherForecast": {
            "new": env.weather_store.handle_weatherForecast_new
        },  # not needed
        "org.powertac.common.WeatherForecastPrediction": {
            "new": env.weather_store.handle_weatherForecastPrediction_new
        },
        "org.powertac.common.WeatherReport": {
            "new": env.weather_store.handle_weatherReport_new
        },
        "org.powertac.du.DefaultBroker": None,
        "org.powertac.du.DefaultBrokerService": None,
        "org.powertac.evcustomer.customers.EvCustomer": None,
    }
