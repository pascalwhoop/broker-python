class Tariff:
    """
    Related to this https://github.com/powertac/powertac-server/wiki/Tariff-representation

    """

    def __init__(self, start, finish, _id, broker_id, power_type , min_duration, signup_payment,
                 early_withdraw_payment,  periodic_payment):
        """
            From the JAVA_DOCS
             * State log fields for readResolve():<br>
             * <code>long brokerId, PowerType powerType, long minDuration,<br>
             * &nbsp;&nbsp;double signupPayment, double earlyWithdrawPayment,<br>
             * &nbsp;&nbsp;double periodicPayment, List<tariffId> supersedes</code>


        :param start:
        :param finish:
        :param _type:
        :param periodic_payment:
        """
        self.start = start
        self.finish = finish
        self.id = _id

        self.broker_id = broker_id
        self.power_type = power_type
        self.min_duration = min_duration
        self.signup_payment = signup_payment
        self.early_withdraw_payment = early_withdraw_payment
        self.periodic_payment = periodic_payment



    @staticmethod
    def from_state_line(line, start, finish):
        parts = line.split("::")

        return Tariff(start, finish, parts[1], parts[3], parts[4], int(parts[5]), round(float(parts[6]),3), round(float(parts[7]),3), round(float(parts[8]),3))