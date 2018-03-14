import util.powertac_communication as comm


comm.connect()

comm.put('<order id="200000181" timeslot="383" mWh="56.49883376466506" limitPrice="-24.152165920135392"><broker>slytherin_v1</broker></order>')

while True:
    msg = comm.get()
    print(msg)
