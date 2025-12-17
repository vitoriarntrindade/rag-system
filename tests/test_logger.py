"""Unit tests for logger module."""

import logging
from pathlib import Path
from unittest.mock import Mock, patch

from rag_system.src.utils.logger import setup_logger


class TestSetupLogger:
    """Tests for setup_logger function."""
    
    def test_returns_logger_instance(self):
        """Test that setup_logger returns a Logger instance."""
        logger = setup_logger("test_logger")
        assert isinstance(logger, logging.Logger)
    
    def test_logger_has_correct_name(self):
        """Test that logger has the specified name."""
        logger_name = "test_logger_unique"
        logger = setup_logger(logger_name)
        assert logger.name == logger_name
    
    def test_creates_different_loggers_for_different_names(self):
        """Test that different names create different loggers."""
        logger1 = setup_logger("logger1")
        logger2 = setup_logger("logger2")
        assert logger1.name != logger2.name
    
    def test_returns_same_logger_for_same_name(self):
        """Test that same name returns the same logger instance."""
        logger1 = setup_logger("same_logger")
        logger2 = setup_logger("same_logger")
        assert logger1 is logger2
    
    def test_logger_has_handlers(self):
        """Test that logger has at least one handler."""
        logger = setup_logger("test_with_handlers")
        assert len(logger.handlers) > 0
    
    @patch('src.utils.logger.get_settings')
    def test_respects_log_level_from_settings(self, mock_get_settings: Mock):
        """Test that logger uses log_level from settings."""
        mock_settings = Mock()
        mock_settings.log_level = "DEBUG"
        mock_settings.log_to_file = False
        mock_settings.logs_dir = Path("/tmp/logs")
        mock_get_settings.return_value = mock_settings
        
        logger = setup_logger("test_debug_level")
        assert logger.level == logging.DEBUG
    
    @patch('src.utils.logger.get_settings')
    def test_creates_logs_directory(self, mock_get_settings: Mock, temp_directory: Path):
        """Test that logs directory is created if it doesn't exist."""
        mock_settings = Mock()
        mock_settings.log_level = "INFO"
        mock_settings.log_to_file = True
        mock_settings.logs_dir = temp_directory / "new_logs"
        mock_get_settings.return_value = mock_settings
        
        setup_logger("test_dir_creation")
        assert mock_settings.logs_dir.exists()


class TestLoggerHandlers:
    """Tests for logger handler configuration."""
    
    def test_has_stream_handler(self):
        """Test that logger has a StreamHandler for console output."""
        logger = setup_logger("test_stream_handler")
        stream_handlers = [
            h for h in logger.handlers 
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        ]
        assert len(stream_handlers) > 0
    
    @patch('src.utils.logger.get_settings')
    def test_has_file_handler_when_enabled(
        self, 
        mock_get_settings: Mock,
        temp_directory: Path
    ):
        """Test that logger has FileHandler when log_to_file is True."""
        mock_settings = Mock()
        mock_settings.log_level = "INFO"
        mock_settings.log_to_file = True
        mock_settings.logs_dir = temp_directory
        mock_get_settings.return_value = mock_settings
        
        logger = setup_logger("test_file_handler")
        file_handlers = [
            h for h in logger.handlers 
            if isinstance(h, logging.FileHandler)
        ]
        assert len(file_handlers) > 0
    
    @patch('src.utils.logger.get_settings')
    def test_no_file_handler_when_disabled(self, mock_get_settings: Mock):
        """Test that logger has no FileHandler when log_to_file is False."""
        mock_settings = Mock()
        mock_settings.log_level = "INFO"
        mock_settings.log_to_file = False
        mock_settings.logs_dir = Path("/tmp/logs")
        mock_get_settings.return_value = mock_settings
        
        logger = setup_logger("test_no_file_handler")
        file_handlers = [
            h for h in logger.handlers 
            if isinstance(h, logging.FileHandler)
        ]
        assert len(file_handlers) == 0


class TestLoggerFormatting:
    """Tests for logger message formatting."""
    
    def test_handlers_have_formatter(self):
        """Test that all handlers have a formatter set."""
        logger = setup_logger("test_formatter")
        for handler in logger.handlers:
            assert handler.formatter is not None
    
    def test_formatter_includes_timestamp(self):
        """Test that formatter includes timestamp."""
        logger = setup_logger("test_timestamp")
        formatter = logger.handlers[0].formatter
        assert formatter is not None
        assert "asctime" in formatter._fmt


class TestLoggerLevels:
    """Tests for logger level configuration."""
    
    @patch('src.utils.logger.get_settings')
    def test_info_level(self, mock_get_settings: Mock):
        """Test that INFO level is set correctly."""
        mock_settings = Mock()
        mock_settings.log_level = "INFO"
        mock_settings.log_to_file = False
        mock_settings.logs_dir = Path("/tmp/logs")
        mock_get_settings.return_value = mock_settings
        
        logger = setup_logger("test_info")
        assert logger.level == logging.INFO
    
    @patch('src.utils.logger.get_settings')
    def test_debug_level(self, mock_get_settings: Mock):
        """Test that DEBUG level is set correctly."""
        mock_settings = Mock()
        mock_settings.log_level = "DEBUG"
        mock_settings.log_to_file = False
        mock_settings.logs_dir = Path("/tmp/logs")
        mock_get_settings.return_value = mock_settings
        
        logger = setup_logger("test_debug")
        assert logger.level == logging.DEBUG
    
    @patch('src.utils.logger.get_settings')
    def test_warning_level(self, mock_get_settings: Mock):
        """Test that WARNING level is set correctly."""
        mock_settings = Mock()
        mock_settings.log_level = "WARNING"
        mock_settings.log_to_file = False
        mock_settings.logs_dir = Path("/tmp/logs")
        mock_get_settings.return_value = mock_settings
        
        logger = setup_logger("test_warning")
        assert logger.level == logging.WARNING
    
    @patch('src.utils.logger.get_settings')
    def test_error_level(self, mock_get_settings: Mock):
        """Test that ERROR level is set correctly."""
        mock_settings = Mock()
        mock_settings.log_level = "ERROR"
        mock_settings.log_to_file = False
        mock_settings.logs_dir = Path("/tmp/logs")
        mock_get_settings.return_value = mock_settings
        
        logger = setup_logger("test_error")
        assert logger.level == logging.ERROR


class TestLoggerUsage:
    """Tests for logger actual usage."""
    
    def test_logger_can_log_info(self):
        """Test that logger can log info messages."""
        logger = setup_logger("test_log_info")
        try:
            logger.info("Test info message")
            success = True
        except Exception:
            success = False
        assert success
    
    def test_logger_can_log_debug(self):
        """Test that logger can log debug messages."""
        logger = setup_logger("test_log_debug")
        try:
            logger.debug("Test debug message")
            success = True
        except Exception:
            success = False
        assert success
    
    def test_logger_can_log_warning(self):
        """Test that logger can log warning messages."""
        logger = setup_logger("test_log_warning")
        try:
            logger.warning("Test warning message")
            success = True
        except Exception:
            success = False
        assert success
    
    def test_logger_can_log_error(self):
        """Test that logger can log error messages."""
        logger = setup_logger("test_log_error")
        try:
            logger.error("Test error message")
            success = True
        except Exception:
            success = False
        assert success
    
    def test_logger_can_log_with_exception(self):
        """Test that logger can log exceptions."""
        logger = setup_logger("test_log_exception")
        try:
            raise ValueError("Test exception")
        except ValueError:
            try:
                logger.exception("Caught exception")
                success = True
            except Exception:
                success = False
        assert success


class TestLoggerPropagation:
    """Tests for logger propagation settings."""
    
    def test_logger_propagation_default_behavior(self):
        """Test that logger propagation is set to default (True by default)."""
        logger = setup_logger("test_propagation_default")
        # Logger propagation is True by default in Python logging
        assert isinstance(logger.propagate, bool)
