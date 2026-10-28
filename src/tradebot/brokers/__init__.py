# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
from tradebot.brokers import crypto, stocks  # noqa: F401  (registers built-ins)
from tradebot.brokers.base import BROKERS, BaseBroker, ManualExecutionBroker, OrderIntent, ReadOnlyBrokerError


def create_broker(name: str, **kwargs) -> BaseBroker:
    """Instantiate a registered broker with credentials from the user config."""
    from tradebot.config import broker_credentials

    kwargs.setdefault("credentials", broker_credentials(name))
    return BROKERS.get(name)(**kwargs)


__all__ = ["BROKERS", "BaseBroker", "ManualExecutionBroker", "OrderIntent", "ReadOnlyBrokerError", "create_broker"]
