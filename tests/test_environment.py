import unittest
import model.environment as env
import tests.test_state_extractor as state_extractor
from model.tariff import Status


class TestEnvironment(unittest.TestCase):


    def test_add_tariff_from_state_line(self):
        pass


    def test_handle_tariff_status_from_state_line(self):
        pass


    def test_handle_tariff_revoke_from_state_line(self):
        spec = "79053:org.powertac.common.TariffSpecification::200000263::-rr::4818::CONSUMPTION::0::0.0::0.0::0.0::null"
        revoke = "194951:org.powertac.common.msg.TariffRevoke::200000429::-rr::4818::200000263"
        env.add_tariff_from_state_line(spec)
        self.assertEqual(1, len(env.tariffs))
        self.assertEqual(env.tariffs["200000263"].status, Status.PENDING)
        env.handle_tariff_revoke_from_state_line(revoke)
        self.assertEqual(env.tariffs["200000263"].status, Status.WITHDRAWN)


    def test_handle_rate_from_state_line(self):
        env.handle_rate_from_state_line(line)

