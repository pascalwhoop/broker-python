import unittest

from model.Tariff import Tariff

lines = [
"5551284:org.powertac.common.TariffSpecification::300007040::-rr::4807::CONSUMPTION::154800000::1.0::-3.0::-1.5::(300006850)",
"5560765:org.powertac.common.TariffSpecification::701127025::-rr::4803::CONSUMPTION::151200000::1.5::-9.0::-1.6116183787035878::null",
"5623199:org.powertac.common.TariffSpecification::200118351::-rr::4806::CONSUMPTION::0::0.0::0.0::0.0::(200092660)",
"5725790:org.powertac.common.TariffSpecification::900171017::-rr::4805::CONSUMPTION::216000000::93.26892692325896::-113.09673619867812::0.0::null",
"5725804:org.powertac.common.TariffSpecification::900171026::-rr::4805::CONSUMPTION::216000000::93.26892692325896::-113.09673619867812::0.0::null",
"5725805:org.powertac.common.TariffSpecification::900171014::-rr::4805::CONSUMPTION::216000000::93.26892692325896::-113.09673619867812::0.0::null",
"5743583:org.powertac.common.TariffSpecification::200119692::-rr::4806::CONSUMPTION::0::0.0::0.0::0.0::(200118351)",
"5770756:org.powertac.common.TariffSpecification::701128054::-rr::4803::CONSUMPTION::151200000::0.5::-8.0::-1.984771095887802::null",
"5980770:org.powertac.common.TariffSpecification::701129156::-rr::4803::CONSUMPTION::151200000::1.0::-8.5::-1.478601111974887::null",
"6031294:org.powertac.common.TariffSpecification::300007666::-rr::4807::CONSUMPTION::154800000::1.0::-3.0::-1.5::null",
"6190780:org.powertac.common.TariffSpecification::701130306::-rr::4803::CONSUMPTION::151200000::2.0::-9.0::-2.1528472327806365::null"
]

class TestTariff(unittest.TestCase):

    def test_from_state_line(self):
        tariffs = [Tariff.from_state_line(l, 0, 100) for l in lines]
        print(tariffs)
        self.assertEqual(-1.5, tariffs[0].periodic_payment)
        self.assertEqual(0.0, tariffs[2].signup_payment)
        self.assertEqual(93.269, tariffs[3].signup_payment)
        self.assertEqual("701130306", tariffs[-1].id)

