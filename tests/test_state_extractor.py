import importlib
import unittest
from unittest.mock import Mock

import statefiles.state_extractor as se


class TestStateExtractor(unittest.TestCase):

    def tearDown(self):
        #because we mocked some functions of an external module, let's reload it so that it
        #doesn't interfere with other tests
        import env.environment as e
        importlib.reload(e)


    def test_parse_state_lines(self):
        # mocking a dependency
        import env.environment as _env
        #mocking our state_handlers away
        _env.handle_tariff_rr = Mock()
        _env.handle_customerInfo = Mock()
        _env.handle_rate_rr = Mock()
        _env.handle_tariffRevoke_new = Mock()
        importlib.reload(se)  # we need to reload this module because during parsing of it, the above function was
        # linked to a local variable
        self.assertEqual(se.environment, _env)

        se.parse_state_lines(msgs)
        self.assertEqual(13, _env.handle_tariff_rr.call_count)
        self.assertEqual(5, _env.handle_customerInfo.call_count)
        self.assertEqual(31, _env.handle_rate_rr.call_count)
        self.assertEqual(2, _env.handle_tariffRevoke_new.call_count)


        # testing for everything else being ignored
        # self.assertEqual(15, len(se.ignored_states))
        # self.assertEqual(-1, _env.tariffs['501597406'].finish)
        # self.assertEqual(29, _env.tariffs['200000263'].finish)

    def test_get_origin(self):
        origin = se.get_class(msgs[0])
        self.assertEqual("org.powertac.common.Competition", origin)
        origin = se.get_class(msgs[5])
        self.assertEqual("org.powertac.common.Rate", origin)


msgs = ['4629:org.powertac.common.Competition::0::withSimulationRate::720\n',
        '4655:org.powertac.common.TimeService::null::setCurrentTime::2013-06-23T00:00:00.000Z\n',
        '5241:org.powertac.common.TimeService::null::setCurrentTime::2013-06-23T00:00:00.000Z\n',
        '6241:org.powertac.common.TimeService::null::setCurrentTime::2013-07-08T00:00:00.000Z\n',
        '6756:org.powertac.common.TariffSpecification::1878::new::1876::CONSUMPTION\n',
        '6757:org.powertac.common.Rate::1879::new\n',
        '6757:org.powertac.common.Rate::1879::withValue::-0.5\n',
        '6757:org.powertac.common.Rate::1879::setTariffId::1878\n',
        '6757:org.powertac.common.TariffSpecification::1878::addRate::1879\n',
        '6758:org.powertac.common.TariffSpecification::1881::new::1876::PRODUCTION\n',
        '6758:org.powertac.common.Rate::1882::new\n',
        '6758:org.powertac.common.Rate::1882::withValue::0.01\n',
        '6758:org.powertac.common.Rate::1882::setTariffId::1881\n',
        '6758:org.powertac.common.TariffSpecification::1881::addRate::1882\n',
        '6758:org.powertac.common.TariffSpecification::1884::new::1876::STORAGE\n',
        '6758:org.powertac.common.Rate::1885::new\n',
        '6758:org.powertac.common.Rate::1885::withValue::-0.5\n',
        '6758:org.powertac.common.Rate::1885::setTariffId::1884\n',
        '6758:org.powertac.common.TariffSpecification::1884::addRate::1885\n',
        '6758:org.powertac.common.RegulationRate::1887::new\n',
        '6758:org.powertac.common.RegulationRate::1887::withUpRegulationPayment::0.01\n',
        '6758:org.powertac.common.RegulationRate::1887::withDownRegulationPayment::-0.5\n',
        '6758:org.powertac.common.RegulationRate::1887::setTariffId::1884\n',
        '6758:org.powertac.common.TariffSpecification::1884::addRate::1887\n',
        '74289:org.powertac.common.TimeService::null::setCurrentTime::2013-07-08T00:00:00.000Z\n',
        '75887:org.powertac.common.RegulationRate::501597405::-rr::501597402::MINUTES::-0.15739769440105333::0.05223975202103926\n',
        '75889:org.powertac.common.Rate::501597403::-rr::501597402::-1::-1::-1::-1::0.0::true::-0.13568766758711495::0.0::0::0.0::0.0\n',
        '75891:org.powertac.common.TariffSpecification::501597402::-rr::4819::ELECTRIC_VEHICLE::0::0.0::0.0::0.0::null\n',
        '75892:org.powertac.common.Rate::501597407::-rr::501597406::-1::-1::-1::-1::0.0::true::-0.15076407509679438::0.0::0::0.0::0.0\n',
        '75892:org.powertac.common.TariffSpecification::501597406::-rr::4819::CONSUMPTION::0::0.0::0.0::-0.95::null\n',
        '75893:org.powertac.common.Rate::401307788::-rr::401307787::-1::-1::18::20::0.0::true::-0.14504015358028285::0.0::0::0.0::0.0\n',
        '75894:org.powertac.common.Rate::401307790::-rr::401307787::-1::-1::6::9::0.0::true::-0.07945860777746452::0.0::0::0.0::0.0\n',
        '75895:org.powertac.common.Rate::401307804::-rr::401307787::-1::-1::21::1::0.0::true::-0.12581381003560116::0.0::0::0.0::0.0\n',
        '75895:org.powertac.common.TariffSpecification::401307787::-rr::4817::CONSUMPTION::0::0.0::0.0::0.0::null\n',
        '75896:org.powertac.common.Rate::401307807::-rr::401307806::-1::-1::-1::-1::0.0::true::-0.09008205947029177::0.0::0::0.0::0.0\n',
        '75897:org.powertac.common.TariffSpecification::401307806::-rr::4817::CONSUMPTION::0::0.0::0.0::0.0::null\n',
        '75897:org.powertac.common.Rate::401307810::-rr::401307809::-1::-1::-1::-1::0.0::true::0.035::0.0::0::0.0::0.0\n',
        '75898:org.powertac.common.TariffSpecification::401307809::-rr::4817::PRODUCTION::0::0.0::0.0::0.0::null\n',
        '134787:org.powertac.common.TariffSpecification::501597481::-rr::4819::CONSUMPTION::0::0.0::0.0::0.0::null\n',
        '184290:org.powertac.common.TimeService::null::setCurrentTime::2013-07-08T22:00:00.000Z\n',
        '189289:org.powertac.common.TimeService::null::setCurrentTime::2013-07-08T23:00:00.000Z\n',
        '194289:org.powertac.common.TimeService::null::setCurrentTime::2013-07-09T00:00:00.000Z\n',
        '194949:org.powertac.common.Rate::200000419::-rr::200000418::-1::-1::0::3::0.0::true::-0.08016581600947167::0.0::0::0.0::0.0\n',
        '194949:org.powertac.common.Rate::200000421::-rr::200000418::-1::-1::4::7::0.0::true::-0.17762851283986683::0.0::0::0.0::0.0\n',
        '194949:org.powertac.common.Rate::200000423::-rr::200000418::-1::-1::8::14::0.0::true::-0.08016581600947167::0.0::0::0.0::0.0\n',
        '194949:org.powertac.common.Rate::200000425::-rr::200000418::-1::-1::15::22::0.0::true::-0.17762851283986683::0.0::0::0.0::0.0\n',
        '194949:org.powertac.common.Rate::200000427::-rr::200000418::-1::-1::23::23::0.0::true::-0.08016581600947167::0.0::0::0.0::0.0\n',
        '194949:org.powertac.common.TariffSpecification::200000418::-rr::4818::CONSUMPTION::0::0.0::0.0::0.0::200000263\n',
        '194951:org.powertac.common.msg.TariffRevoke::200000429::-rr::4818::200000263\n',
        '199289:org.powertac.common.TimeService::null::setCurrentTime::2013-07-09T01:00:00.000Z\n',
        '199423:org.powertac.common.msg.TariffRevoke::63237::new::4818::200000263\n',
        '204289:org.powertac.common.TimeService::null::setCurrentTime::2013-07-09T02:00:00.000Z\n',
        '209289:org.powertac.common.TimeService::null::setCurrentTime::2013-07-09T03:00:00.000Z\n',
        '214289:org.powertac.common.TimeService::null::setCurrentTime::2013-07-09T04:00:00.000Z\n',
        '219289:org.powertac.common.TimeService::null::setCurrentTime::2013-07-09T05:00:00.000Z\n',
        '224289:org.powertac.common.TimeService::null::setCurrentTime::2013-07-09T06:00:00.000Z\n',
        '229289:org.powertac.common.TimeService::null::setCurrentTime::2013-07-09T07:00:00.000Z\n',
        '234290:org.powertac.common.TimeService::null::setCurrentTime::2013-07-09T08:00:00.000Z\n',
        '239289:org.powertac.common.TimeService::null::setCurrentTime::2013-07-09T09:00:00.000Z\n',
        '244290:org.powertac.common.TimeService::null::setCurrentTime::2013-07-09T10:00:00.000Z\n',
        '245078:org.powertac.common.Rate::300000124::-rr::300000126::-1::-1::-1::-1::0.0::true::-0.16::0.0::0::0.0::0.0\n',
        '245078:org.powertac.common.TariffSpecification::300000126::-rr::4816::CONSUMPTION::1209600000::10.0::-10.0::-1.5::null\n',
        '249289:org.powertac.common.TimeService::null::setCurrentTime::2013-07-09T11:00:00.000Z\n',
        '254289:org.powertac.common.TimeService::null::setCurrentTime::2013-07-09T12:00:00.000Z\n',
        '254823:org.powertac.common.Rate::501598002::-rr::501598001::-1::-1::-1::-1::0.0::true::-0.1343307909112438::0.0::0::0.0::0.0\n',
        '254823:org.powertac.common.TariffSpecification::501598001::-rr::4819::CONSUMPTION::0::0.0::0.0::0.0::null\n',
        '259289:org.powertac.common.TimeService::null::setCurrentTime::2013-07-09T13:00:00.000Z\n',
        '264289:org.powertac.common.TimeService::null::setCurrentTime::2013-07-09T14:00:00.000Z\n',
        '269290:org.powertac.common.TimeService::null::setCurrentTime::2013-07-09T15:00:00.000Z\n',
        '274290:org.powertac.common.TimeService::null::setCurrentTime::2013-07-09T16:00:00.000Z\n',
        '279289:org.powertac.common.TimeService::null::setCurrentTime::2013-07-09T17:00:00.000Z\n',
        '284290:org.powertac.common.TimeService::null::setCurrentTime::2013-07-09T18:00:00.000Z\n',
        '284851:org.powertac.common.Rate::700861939::-rr::700861938::1::5::0::1::0.0::true::-0.1348102653122821::0.0::0::0.0::0.0\n',
        '284851:org.powertac.common.Rate::700861941::-rr::700861938::6::7::0::1::0.0::true::-0.02638934592333807::0.0::0::0.0::0.0\n',
        '284852:org.powertac.common.Rate::700861943::-rr::700861938::1::5::1::2::0.0::true::-0.12579529752613972::0.0::0::0.0::0.0\n',
        '284857:org.powertac.common.Rate::700862025::-rr::700861938::6::7::21::22::0.0::true::-0.034821092773111254::0.0::0::0.0::0.0\n',
        '284857:org.powertac.common.Rate::700862027::-rr::700861938::1::5::22::23::0.0::true::-0.22831710124561203::0.0::0::0.0::0.0\n',
        '284857:org.powertac.common.Rate::700862029::-rr::700861938::6::7::22::23::0.0::true::-0.03114335269794573::0.0::0::0.0::0.0\n',
        '284857:org.powertac.common.Rate::700862031::-rr::700861938::1::5::23::0::0.0::true::-0.14031099887382975::0.0::0::0.0::0.0\n',
        '284857:org.powertac.common.Rate::700862033::-rr::700861938::6::7::23::0::0.0::true::-0.028205917617700346::0.0::0::0.0::0.0\n',
        '284857:org.powertac.common.TariffSpecification::700861938::-rr::4812::CONSUMPTION::151200000::1.5::-2.5::-0.6772992931665452::null\n',
        '284858:org.powertac.common.msg.TariffRevoke::700862035::-rr::4812::700861394\n',
        '289290:org.powertac.common.TimeService::null::setCurrentTime::2013-07-09T19:00:00.000Z\n',
        '289416:org.powertac.common.msg.TariffRevoke::102101::new::4812::700861394\n',
        '294289:org.powertac.common.TimeService::null::setCurrentTime::2013-07-09T20:00:00.000Z\n',
        '299289:org.powertac.common.TimeService::null::setCurrentTime::2013-07-09T21:00:00.000Z\n',
        '304289:org.powertac.common.TimeService::null::setCurrentTime::2013-07-09T22:00:00.000Z\n',
        '309289:org.powertac.common.TimeService::null::setCurrentTime::2013-07-09T23:00:00.000Z\n',
        '314289:org.powertac.common.TimeService::null::setCurrentTime::2013-07-10T00:00:00.000Z\n',
        '319289:org.powertac.common.TimeService::null::setCurrentTime::2013-07-10T01:00:00.000Z\n',
        '324289:org.powertac.common.TimeService::null::setCurrentTime::2013-07-10T02:00:00.000Z\n',
        '329289:org.powertac.common.TimeService::null::setCurrentTime::2013-07-10T03:00:00.000Z\n',
        '334289:org.powertac.common.TimeService::null::setCurrentTime::2013-07-10T04:00:00.000Z\n',
        '339289:org.powertac.common.TimeService::null::setCurrentTime::2013-07-10T05:00:00.000Z\n',
        '344289:org.powertac.common.TimeService::null::setCurrentTime::2013-07-10T06:00:00.000Z\n',
        '349289:org.powertac.common.TimeService::null::setCurrentTime::2013-07-10T07:00:00.000Z\n',
        '354289:org.powertac.common.TimeService::null::setCurrentTime::2013-07-10T08:00:00.000Z\n',
        '359289:org.powertac.common.TimeService::null::setCurrentTime::2013-07-10T09:00:00.000Z\n',
        '364289:org.powertac.common.TimeService::null::setCurrentTime::2013-07-10T10:00:00.000Z\n',
        '365138:org.powertac.common.Rate::300000290::-rr::300000292::-1::-1::-1::-1::0.0::true::0.0432::0.0::0::0.0::0.0\n',
        '365139:org.powertac.common.TariffSpecification::300000292::-rr::4816::WIND_PRODUCTION::604800000::1.0::-0.8::-0.05::null\n',
        '369290:org.powertac.common.TimeService::null::setCurrentTime::2013-07-10T11:00:00.000Z\n',
        '374290:org.powertac.common.TimeService::null::setCurrentTime::2013-07-10T12:00:00.000Z\n',
        '379289:org.powertac.common.TimeService::null::setCurrentTime::2013-07-10T13:00:00.000Z\n',
        '384289:org.powertac.common.TimeService::null::setCurrentTime::2013-07-10T14:00:00.000Z\n',
        '389289:org.powertac.common.TimeService::null::setCurrentTime::2013-07-10T15:00:00.000Z\n',
        '394289:org.powertac.common.TimeService::null::setCurrentTime::2013-07-10T16:00:00.000Z\n',
        '399289:org.powertac.common.TimeService::null::setCurrentTime::2013-07-10T17:00:00.000Z\n',
        '404289:org.powertac.common.TimeService::null::setCurrentTime::2013-07-10T18:00:00.000Z\n',
        '725273:org.powertac.common.Rate::300000818::-rr::300000821::-1::-1::-1::-1::0.0::true::-0.135424::0.0::0::0.0::0.0\n',
        '725273:org.powertac.common.TariffSpecification::300000821::-rr::4816::CONSUMPTION::1209600000::10.0::-10.0::-1.5::null\n',
        '729289:org.powertac.common.TimeService::null::setCurrentTime::2013-07-13T11:00:00.000Z\n',
        '734289:org.powertac.common.TimeService::null::setCurrentTime::2013-07-13T12:00:00.000Z\n',
        '739290:org.powertac.common.TimeService::null::setCurrentTime::2013-07-13T13:00:00.000Z\n',
        '744290:org.powertac.common.TimeService::null::setCurrentTime::2013-07-13T14:00:00.000Z\n',
        '744656:org.powertac.common.Rate::501599942::-rr::501599941::-1::-1::-1::-1::0.0::true::-0.1343307909112438::0.0::0::0.0::0.0\n',
        '744656:org.powertac.common.TariffSpecification::501599941::-rr::4819::CONSUMPTION::0::0.0::0.0::0.0::null\n',
        '749290:org.powertac.common.TimeService::null::setCurrentTime::2013-07-13T15:00:00.000Z\n',
        '754289:org.powertac.common.TimeService::null::setCurrentTime::2013-07-13T16:00:00.000Z\n',
        '1754712:org.powertac.common.Rate::700868499::-rr::700868494::1::5::1::2::0.0::true::-0.04515193018584082::0.0::0::0.0::0.0\n',
        '1754713:org.powertac.common.Rate::700868501::-rr::700868494::6::7::1::2::0.0::true::-0.034227609552561114::0.0::0::0.0::0.0\n',
        '1754721:org.powertac.common.Rate::700868583::-rr::700868494::1::5::22::23::0.0::true::-0.06435845745953268::0.0::0::0.0::0.0\n',
        '1754721:org.powertac.common.Rate::700868585::-rr::700868494::6::7::22::23::0.0::true::-0.0389584684413752::0.0::0::0.0::0.0\n',
        '1754722:org.powertac.common.Rate::700868587::-rr::700868494::1::5::23::0::0.0::true::-0.057414758382329546::0.0::0::0.0::0.0\n',
        '1754727:org.powertac.common.Rate::700868589::-rr::700868494::6::7::23::0::0.0::true::-0.03296667938900334::0.0::0::0.0::0.0\n',
        '10160:org.powertac.common.CustomerInfo::4736::withPowerType::ELECTRIC_VEHICLE',
        '10160:org.powertac.common.CustomerInfo::4736::withControllableKW::-6.6',
        '10160:org.powertac.common.CustomerInfo::4736::withUpRegulationKW::-6.6',
        '10160:org.powertac.common.CustomerInfo::4736::withDownRegulationKW::6.6',
        '10160:org.powertac.common.CustomerInfo::4736::withStorageCapacity::24.0'
       ]
