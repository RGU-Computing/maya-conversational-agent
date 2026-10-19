"""Model requests use the deployment's explicit timeout/retry settings."""

import os
import unittest
from unittest.mock import patch

from maya.framework.integrations import models


class ModelConfigurationTests(unittest.TestCase):
    def test_timeout_and_retries_are_configured_and_client_is_reused(self):
        models.get_chat_model.cache_clear()
        self.addCleanup(models.get_chat_model.cache_clear)
        with patch.dict(os.environ, {
            "AZURE_OPENAI_DEPLOYMENT": "offline", "AZURE_OPENAI_API_VERSION": "offline",
            "AZURE_OPENAI_ENDPOINT": "https://example.invalid",
            "MAYA_MODEL_TIMEOUT_SECONDS": "12.5", "MAYA_MODEL_MAX_RETRIES": "0",
        }), patch.object(models, "load_dotenv"), patch.object(models, "AzureChatOpenAI") as constructor:
            self.assertIs(models.get_chat_model(), models.get_chat_model())
            constructor.assert_called_once()
            self.assertEqual(constructor.call_args.kwargs["timeout"], 12.5)
            self.assertEqual(constructor.call_args.kwargs["max_retries"], 0)
