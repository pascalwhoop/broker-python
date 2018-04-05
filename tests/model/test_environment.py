import unittest
from datetime import datetime
from unittest.mock import MagicMock

from model.rate import Rate
from model.tariff import Status, Tariff, TariffStats
from model.StatelineParser import StatelineParser
import model.tariff_transaction as tt
import env.environment as env
import model.customer_info as ci
import tests.teststrings as strings


class TestEnvironment(unittest.TestCase):

    def setUp(self):
        env.reset()

    def test_handle_tariff_rr(self):
        line = '75891:org.powertac.common.TariffSpecification::501597402::-rr::4819::ELECTRIC_VEHICLE::0::0.0::0.0::0.0::null\n'
        # TODO do we also need to handle new specs? They get "newed" if they are from the default broker right?"
        env.handle_tariff_rr(line)
        self.assertEqual(env.tariffs["501597402"].powerType, "ELECTRIC_VEHICLE")

    def test_handle_tariff_status_from_state_line(self):
        pass

    def test_handle_tariff_revoke_from_state_line(self):
        spec = "79053:org.powertac.common.TariffSpecification::200000263::-rr::4818::CONSUMPTION::0::0.0::0.0::0.0::null"
        revoke = "194951:org.powertac.common.msg.TariffRevoke::200000429::-rr::4818::200000263"
        env.handle_tariff_rr(spec)
        self.assertEqual(1, len(env.tariffs))
        self.assertEqual(env.tariffs["200000263"].status, Status.PENDING)
        env.handle_tariffRevoke_new(revoke)
        self.assertEqual(env.tariffs["200000263"].status, Status.WITHDRAWN)

    def test_handle_rate_from_state_line(self):
        # env.handle_rate_from_state_line(line)
        pass

    def test_handle_customerInfo(self):
        ci.CustomerInfo._jsm_withPowerType = MagicMock()
        ci.CustomerInfo._jsm_withUpRegulationKW = MagicMock()

        customer_info_lines = strings.STATE_LINES[1:6]
        for l in customer_info_lines:
            env.handle_customerInfo(l)
        self.assertEqual(1, ci.CustomerInfo._jsm_withPowerType.call_count)
        ci.CustomerInfo._jsm_withUpRegulationKW.assert_called_with("-6.6")

    def test_get_tariff_type(self):
        line = "8828:org.powertac.common.TariffTransaction::3328::new::1876::360::SIGNUP::1878::3320::30000::0.0::-0.0::false"
        parts = StatelineParser.split_line(line)
        t = env._get_tariff_type(parts)
        self.assertEqual(tt.TransactionType.SIGNUP, t)

    def test_handle_transaction_consume(self):
        # test data
        consume_line = "100308:org.powertac.common.TariffTransaction::5495::new::1876::360::CONSUME::1878::3735::1::-4.444444444444445::2.2222222222222223::false"
        tariff_id = "1878"
        # mock data
        env.tariffs[tariff_id] = Tariff(id_=tariff_id)
        stats = TariffStats(0)
        stats.initial_timeslot = 360
        env.tariff_stats[tariff_id] = stats
        # call
        env.handle_TariffTransaction_new(consume_line)
        # asserts
        # assert transactions is stored locally
        self.assertIsNotNone(env.transactions["3735"][0])
        # assert stats are updated accordingly
        self.assertEqual(-4.4444, stats.timeslot_stats[0][5])
        self.assertEqual(2.2222, stats.timeslot_stats[0][2])
        # let's add another one
        consume_line = "100308:org.powertac.common.TariffTransaction::5495::new::1876::360::CONSUME::1878::3735::1::-2.234444444445::5.1235222222222223::false"
        env.handle_TariffTransaction_new(consume_line)
        # assert sums make sense
        self.assertEqual(-6.6788, stats.timeslot_stats[0][5])
        self.assertEqual(7.3457, stats.timeslot_stats[0][2])

        # let's add another one but from a different timeslot (1 later)
        consume_line = "100308:org.powertac.common.TariffTransaction::5495::new::1876::361::CONSUME::1878::3735::1::-2.234444444445::5.1235222222222223::false"
        env.handle_TariffTransaction_new(consume_line)
        # assert sums make sense
        self.assertEqual(2, len(stats.timeslot_stats))
        self.assertEqual(-2.2344, stats.timeslot_stats[1][5])
        self.assertEqual(5.1235, stats.timeslot_stats[1][2])

    def test_add_transaction(self):
        trans = tt.TariffTransaction(customerInfo= "foo")
        env.current_timestep = 1
        env._add_transaction(trans)

        self.assertEqual(env.transactions["foo"][0][0], trans)

    def test_handle_timeslotUpdate_new(self):
        lines = [l for l in strings.STATE_LINES if "TimeslotUpdate" in l]
        env.handle_timeslotUpdate_new(lines[0])

        self.assertEqual(383, env.first_enabled)
        self.assertEqual(406, env.last_enabled)
        self.assertEqual(382, env.current_timestep)

        for l in lines:
            env.handle_timeslotUpdate_new(l)

        self.assertEqual(381+len(lines), env.current_timestep)
        date = datetime(year=2014, month=12, day=27, hour=11) #2014-12-27T11:00:00.000Z
        self.assertEqual(date, env.current_tod)

    def test_handle_weatherForecastPrediction_new(self):
        """test if we parse the forecasts correctly. that is because they all get created at the beginning and it's hard
        to get them ID'd so that they are easily retrievable from a map. we need to parse 2 state lines to get that
        done. 
        1. store weatherPrediction by ID
        2. when Forecasts are built, replace them with a "origin+time" key
        """
        lines = [
            "6428:org.powertac.common.WeatherForecastPrediction::26::new::1::-11.0::20.0::0.0::1.0",
            "6428:org.powertac.common.WeatherForecastPrediction::27::new::1::-11.0::20.0::0.0::1.0",
            "6551:org.powertac.common.WeatherForecastPrediction::242::new::1::-11.0::14.0::0.0::1.0",
            "6551:org.powertac.common.WeatherForecastPrediction::243::new::1::-11.0::14.0::0.0::1.0",
            "6651:org.powertac.common.WeatherForecast::602::new::360::(26,27)",
            "6673:org.powertac.common.WeatherForecast::611::new::369::(242,243)"
            ]
        #adding the predictions to environment
        env.handle_weatherForecastPrediction_new(lines[0])
        env.handle_weatherForecastPrediction_new(lines[1])
        env.handle_weatherForecastPrediction_new(lines[2])
        env.handle_weatherForecastPrediction_new(lines[3])
        self.assertIn("26", env.weather_predictions)
        self.assertIn("242", env.weather_predictions)
        env.handle_weatherForecast_new(lines[4])
        env.handle_weatherForecast_new(lines[5])
        self.assertIn("360+1", env.weather_predictions)
        self.assertIn("369+1", env.weather_predictions)


    def test_handle_weatherReport_new(self):
        lines = [l for l in strings.STATE_LINES if "WeatherReport" in l]
        env.handle_weatherReport_new(lines[0])

        self.assertEqual(1, len(env.weather_reports))

    def test_handle_timeslotUpdate_new2(self):
        line = "4668:org.powertac.common.Competition::0::withSimulationBaseTime::1418256000000"
        env.handle_competition_withSimulationBaseTime(line)
        expected = datetime(2014, 12, 11, 1, 0)

        self.assertEqual(expected, env.first_tod)

    def test_rate_left_over(self):
        # sometimes rates have the same ID. Also, they get added to multiple tariffs under the same ID. That's dirty as hell
        # but we gotta deal with it. So let's test for the case
        # 1. rate 1 creation TariffA
        # 2. rate 1 creation TariffB
        # 3. Tariff creation
        # 4. Rate not being added to Tariff --> still applicable though
        lines = [
            "681592:org.powertac.common.Rate::700111446::-rr::700111445::-1::-1::-1::-1::0.0::true::0.0495::0.0::0::0.0::0.0",
            "681593: org.powertac.common.Rate::700111446::-rr::700111448::-1::-1::-1::-1::0.0::true::0.0495::0.0::0::0.0::0.0",
            "681592:org.powertac.common.TariffSpecification::700111445::-rr::4803::SOLAR_PRODUCTION::432000000::0.0::-10.0::-6.0::null"
        ]
        env.handle_rate_rr(lines[0])
        env.handle_rate_rr(lines[1])
        env.handle_tariff_rr(lines[2])

        self.assertEqual("700111446", env.tariffs["700111445"]._rates[0].id_)



    def test_get_rate_for_customer_transaction(self):
        # testing if the date times are calculated appropriately
        # a powertac day 3 is a wednesday but its a python thursday
        transaction1 = tt.TariffTransaction(tariffSpec="1", when=4+48) #4h after midnight on sunday
        transaction2 = tt.TariffTransaction(tariffSpec="1", when=12+3*24) # noon on monday
        transaction3 = tt.TariffTransaction(tariffSpec="1", when=4+24) # 4h after midnight on satruday
        rate1 = Rate(id_="r1", tariffId="1", weeklyBegin=7, weeklyEnd=7, dailyBegin=4, dailyEnd=7) # sunday nights
        rate2 = Rate(id_="r2", tariffId="1", weeklyBegin=1, weeklyEnd=5, dailyBegin=-1, dailyEnd=-1) #all day weekdays
        rate3 = Rate(id_="r3", tariffId="1", weeklyBegin=6, weeklyEnd=6, dailyBegin=4, dailyEnd=7) #saturday nights

        tariff = Tariff(id_="1")
        tariff.add_rate(rate1)
        tariff.add_rate(rate2)
        tariff.add_rate(rate3)
        env.tariffs[tariff.id_] = tariff

        env.rates[rate1.id_] = rate1
        env.rates[rate2.id_] = rate2
        env.rates[rate3.id_] = rate3
        env.first_tod = datetime(year=2010, month=1, day=1, hour=0, minute=0) #friday
        self.assertEqual(rate1, env.get_rate_for_customer_transactions([transaction1]))
        self.assertEqual(rate2, env.get_rate_for_customer_transactions([transaction2]))
        self.assertEqual(rate3, env.get_rate_for_customer_transactions([transaction3]))

