from app import config
from app.audit_log import AuditLog
from app.model_service import ModelService

service = ModelService(config.MODEL_PATH, config.MODEL_INFO_PATH)
audit = AuditLog(config.AUDIT_DB_PATH)
