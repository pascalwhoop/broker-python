import unittest

from model.rate import Rate
from model.tariff import Tariff

tariff_lines = [
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

rate_lines = [
    "7240745:org.powertac.common.Rate::701136210::-rr::701136153::1::5::14::15::0.0::true::-0.08033842857581179::0.0::0::0.0::0.0",
    "7240746:org.powertac.common.Rate::701136212::-rr::701136153::6::7::14::15::0.0::true::-0.0672139355585116::0.0::0::0.0::0.0",
    "7240746:org.powertac.common.Rate::701136214::-rr::701136153::1::5::15::16::0.0::true::-0.08249962240634248::0.0::0::0.0::0.0",
    "7240746:org.powertac.common.Rate::701136216::-rr::701136153::6::7::15::16::0.0::true::-0.08418327255669648::0.0::0::0.0::0.0",
    "7240746:org.powertac.common.Rate::701136218::-rr::701136153::1::5::16::17::0.0::true::-0.10033446844722532::0.0::0::0.0::0.0",
    "7240746:org.powertac.common.Rate::701136220::-rr::701136153::6::7::16::17::0.0::true::-0.07407212718813878::0.0::0::0.0::0.0",
    "7240746:org.powertac.common.Rate::701136222::-rr::701136153::1::5::17::18::0.0::true::-0.09420013680244543::0.0::0::0.0::0.0",
    "7240746:org.powertac.common.Rate::701136224::-rr::701136153::6::7::17::18::0.0::true::-0.1602196462871132::0.0::0::0.0::0.0",
    "7240746:org.powertac.common.Rate::701136226::-rr::701136153::1::5::18::19::0.0::true::-0.08901279210535747::0.0::0::0.0::0.0",
    "7240746:org.powertac.common.Rate::701136228::-rr::701136153::6::7::18::19::0.0::true::-0.0820396493591319::0.0::0::0.0::0.0",
    "7240746:org.powertac.common.Rate::701136230::-rr::701136153::1::5::19::20::0.0::true::-0.09432226372245299::0.0::0::0.0::0.0",
    "7240747:org.powertac.common.Rate::701136232::-rr::701136153::6::7::19::20::0.0::true::-0.09026078386766792::0.0::0::0.0::0.0",
    "7240747:org.powertac.common.Rate::701136234::-rr::701136153::1::5::20::21::0.0::true::-0.07588429904485892::0.0::0::0.0::0.0",
    "7240747:org.powertac.common.Rate::701136236::-rr::701136153::6::7::20::21::0.0::true::-0.07291721079262241::0.0::0::0.0::0.0",
    "7240747:org.powertac.common.Rate::701136238::-rr::701136153::1::5::21::22::0.0::true::-0.08108145735734072::0.0::0::0.0::0.0",
    "7240747:org.powertac.common.Rate::701136240::-rr::701136153::6::7::21::22::0.0::true::-0.06274933045152474::0.0::0::0.0::0.0",
    "7240747:org.powertac.common.Rate::701136242::-rr::701136153::1::5::22::23::0.0::true::-0.061883678436572505::0.0::0::0.0::0.0",
    "7240747:org.powertac.common.Rate::701136244::-rr::701136153::6::7::22::23::0.0::true::-0.06012183207243654::0.0::0::0.0::0.0",
    "7240747:org.powertac.common.Rate::701136246::-rr::701136153::1::5::23::0::0.0::true::-0.05560073080491179::0.0::0::0.0::0.0",
    "7240747:org.powertac.common.Rate::701136248::-rr::701136153::6::7::23::0::0.0::true::-0.04845575456056679::0.0::0::0.0::0.0",
    "7302693:org.powertac.common.Rate::200140767::-rr::200140766::-1::-1::0::23::0.0::true::-0.09877534421976514::0.0::0::0.0::0.0"]


class TestTariff(unittest.TestCase):

    def test_Tariff_from_state_line(self):
        tariffs = [Tariff.from_state_line(l) for l in tariff_lines]
        print(tariffs)
        self.assertEqual(-1.5, tariffs[0].periodic_payment)
        self.assertEqual(0.0, tariffs[2].signup_payment)
        self.assertEqual(93.269, tariffs[3].signup_payment)
        self.assertEqual("701130306", tariffs[-1].id)

    def test_Rate_from_state_line(self):
        rates = [Rate.from_state_line(l) for l in rate_lines]
        self.assertEqual(14, rates[1].daily_begin)
        self.assertEqual(-0.099, rates[-1].min_value)
