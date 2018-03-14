"""
These functions serve the communication with the server.
Because the java client doesn't help us with ID management and prefixing the xml messages (some form of validation technique used in powertac), we need to do this on our own.
see `PowerTacBroker.java#handleMessage` for details

1. <broker-authentication username="brokername" password="secret" brokerTime="1521043835291"/>  is sent to server (handled by Java Adapter)
2. <broker-accept prefix="2" key="dlop5b" serverTime="1521046957413"/>                          is received by broker (cooked and passed)
3. we need to use the prefix for our ID generation and the key for prefixing xml messages. I know, the names...
4. Keep track of serverTime is also helpful so we can make sure we don't think too long with our agent.

"""
import xml.etree.ElementTree as ET

prefix      = 0 #invalid default value
cnt         = 0
multiplier  = 100000000
key         = "x"


def get_string(_id):
    return str(_id / multiplier) + "." + str(_id % multiplier)


def extract_prefix(_id):
    return round(_id / multiplier)  # need to round


def create_id():
    global cnt
    cnt += 1
    return multiplier * prefix + cnt


def broker_accept_intercept(msg: str):
    import util.powertac_communication as comm #importing inside function because of circular deps
    if msg.startswith("<broker-accept"):
        handle_broker_accept_line(msg)
        # removing itself from interceptors after handling it once
        comm.interceptors.remove(broker_accept_intercept)

def handle_broker_accept_line(line: str):
    line_xml = ET.fromstring(line)

    global prefix
    global key

    prefix = line_xml.attrib["prefix"]
    key = line_xml.attrib["key"]


# explicit setter for prefix
def set_prefix(_prefix):
    global prefix
    prefix = _prefix


# explicit setter for key
def set_key(_key):
    global key
    key = _key
