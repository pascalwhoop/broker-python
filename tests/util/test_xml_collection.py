import unittest
import util.make_xml_collection as mxc
import tests.teststrings  as ts
from bs4 import BeautifulSoup
class TestXmlCollection(unittest.TestCase):
    """test class for make_xml_collection"""

    def test_parse_message(self):
        xml = ts.XML_MESSAGE
        xml_obj = mxc.parse_message(xml)
        tariff_tx = xml_obj.find("tariff-tx")
        self.assertEqual("806408",tariff_tx["id"])
        self.assertEqual("4658", tariff_tx.customerInfo.string)

    def test_add_to_type_set(self):
        #creating first instance of received xml and adding to known xml repo
        xml_obj = mxc.parse_message(ts.XML_MESSAGE)
        mxc.add_to_type_set(xml_obj)
        self.assertEqual(xml_obj, mxc.xml_types.get("tariff-tx"))

        #creating a second one and adding some children to it that need to be recursively merged
        xml_obj2 = mxc.parse_message(ts.XML_MESSAGE)
        chicken = BeautifulSoup("<chicken></chicken>", "xml").findChild()
        chicken['legs'] = "2"
        chicken.append(BeautifulSoup("<eyes><eye1>left</eye1><eye2>right</eye2></eyes>","xml").findChild())
        xml_obj2.findChild().append(chicken)

        xml_obj2.findChild()["milk"] = "true"
        self.assertEqual("[document]", xml_obj2.name)
        mxc.add_to_type_set(xml_obj2)
        self.assertEqual("left", mxc.xml_types["tariff-tx"].chicken.eyes.eye1.string)
        self.assertEqual("true", mxc.xml_types["tariff-tx"].findChild()["milk"])



#<tariff-tx id="806408" postedTimeslot="711" txType="CONSUME" customerCount="1" kWh="-0.0" charge="0.0" regulation="false">
#  <broker>Sample</broker>
#  <customerInfo>4658</customerInfo>
#  <tariffSpec>200000026</tariffSpec>
#</tariff-tx>
