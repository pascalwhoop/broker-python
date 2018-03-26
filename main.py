import util.powertac_communication as comm
import util.make_xml_collection as mxc



def create_sample_xml():
    """Generates a set of sample xml files from a communication session with the server
    """
    comm.connect()
    msg_counter = 0
    completed = False
    while not completed:
        msg = comm.get()
        msg_counter += 1
        xml = mxc.parse_message(msg)
        mxc.add_to_type_set(xml)
        if msg_counter % 100 == 0:
            print("Msg: {}  Known Types: {}".format(msg_counter, len(mxc.xml_types)))

        if msg_counter % 1000 == 0:
            print("pickling")
            mxc.pickle_xml()
