import unittest
from unittest.mock import patch, Mock
import util.config as cfg

from agent_components.tariffs.publisher import TariffPublisher
from communication.grpc_messages_pb2 import PBTariffSpecification, PBPowerType, PBTariffRevoke, PBRate


class TestPublisher(unittest.TestCase):

    def setUp(self):
        self.p = TariffPublisher()

    @patch('agent_components.tariffs.publisher.dispatcher')
    def test_clone(self, dispatcher_mock:Mock):
        enemy_spec = PBTariffSpecification(id=1,
                                           broker=cfg.TARIFF_CLONE_COMPETITOR_AGENT,
                                           expiration=42,
                                           minDuration=42,
                                           powerType=PBPowerType(label="CONSUME"),
                                           rates=[PBRate(tariffId=1)]
                                           )
        self.p.handle_tariff_spec(None, None, enemy_spec)
        mine = dispatcher_mock.send.call_args[1]['msg']
        assert mine.broker == cfg.ME
        assert mine.rates[0].tariffId != 1
        enemy_spec.broker = "micky"
        self.p.handle_tariff_spec(None, None, enemy_spec)
        assert dispatcher_mock.send.call_count == 1

    @patch('agent_components.tariffs.publisher.dispatcher')
    def test_revoke(self, dispatcher_mock:Mock):
        enemy_revoke = PBTariffRevoke(tariffId=1, broker=cfg.TARIFF_CLONE_COMPETITOR_AGENT)
        self.p.handle_tariff_revoke(None, None, enemy_revoke)
        assert dispatcher_mock.send.call_count == 0

        #but with id in clones it's cool
        self.p.clones[1] = 2
        self.p.handle_tariff_revoke(None, None, enemy_revoke)
        assert dispatcher_mock.send.call_count == 1


