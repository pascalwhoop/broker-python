import unittest
from unittest.mock import patch, Mock
import util.config as cfg

from agent_components.tariffs.publisher import TariffPublisher
from communication.grpc_messages_pb2 import PBTariffSpecification, PBPowerType, PBTariffRevoke


class TestPublisher(unittest.TestCase):

    def setUp(self):
        self.p = TariffPublisher()

    @patch('agent_components.tariffs.publisher.submit_service')
    def test_clone(self, submit_mock:Mock):
        enemy_spec = PBTariffSpecification(id=1, broker=cfg.TARIFF_CLONE_COMPETITOR_AGENT, expiration=42,minDuration=42,powerType=PBPowerType(label="CONSUME") )
        self.p.handle_tariff_spec(None, None, enemy_spec)
        mine = submit_mock.send_tariff_spec.call_args[0][0]
        assert mine.broker == cfg.ME
        enemy_spec.broker = "micky"
        self.p.handle_tariff_spec(None, None, enemy_spec)
        assert submit_mock.send_tariff_spec.call_count == 1

    @patch('agent_components.tariffs.publisher.submit_service')
    def test_revoke(self, submit_mock:Mock):
        enemy_revoke = PBTariffRevoke(tariffId=1, broker=cfg.TARIFF_CLONE_COMPETITOR_AGENT)
        self.p.handle_tariff_revoke(None, None, enemy_revoke)
        assert submit_mock.send_tariff_revoke.call_count == 0

        #but with id in clones it's cool
        self.p.clones[1] = 2
        self.p.handle_tariff_revoke(None, None, enemy_revoke)
        assert submit_mock.send_tariff_revoke.call_count == 1


