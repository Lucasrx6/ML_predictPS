"""
========================================
SERVICE WRAPPER — ML PS VOLUME WORKER
========================================

Mantém o worker vivo como serviço Windows via NSSM.
Usa a biblioteca 'schedule' para disparar o main() do worker
todo dia no horário configurado.

Gerenciado por NSSM:
  nssm install ML_PS_Volume C:\ml_workspace\venv_ml\Scripts\python.exe
  nssm set ML_PS_Volume AppParameters C:\ml_workspace\workers\ml_ps_volume_service.py
  nssm set ML_PS_Volume AppDirectory C:\ml_workspace
  nssm start ML_PS_Volume

Logs em: ../logs/ml_ps_volume_service.log
"""
import sys
import time
import logging
from datetime import datetime
from pathlib import Path

import schedule

# Adiciona o diretório atual ao path para importar o worker
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ml_ps_volume_worker import main as worker_main


# ========================================
# CONFIGURAÇÃO
# ========================================

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_PATH = BASE_DIR / 'logs' / 'ml_ps_volume_service.log'

# Horário de execução diária (24h format)
HORARIO_EXECUCAO = "04:30"

# Intervalo de verificação do scheduler (1 minuto é suficiente)
CHECK_INTERVAL_SECONDS = 60

LOG_PATH.parent.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH, encoding='utf-8'),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


def executar_worker_com_protecao():
    """
    Wrapper que captura qualquer exceção do worker.
    Importante: se o worker falhar, o serviço NÃO pode cair —
    ele precisa continuar vivo para tentar de novo amanhã.
    """
    try:
        logger.info(">>> Disparando execução agendada do worker ML PS Volume")
        worker_main()
        logger.info(">>> Execução agendada concluída com sucesso")
    except SystemExit as e:
        # worker_main() chama sys.exit(1) em erro fatal; captura aqui
        logger.error(f">>> Worker terminou com SystemExit: {e}")
    except Exception as e:
        logger.error(f">>> Erro não tratado no worker: {type(e).__name__}: {e}", exc_info=True)


def main():
    logger.info("=" * 60)
    logger.info("ML PS VOLUME SERVICE — INÍCIO")
    logger.info(f"Agendamento: diariamente às {HORARIO_EXECUCAO}")
    logger.info("=" * 60)

    # Agenda execução diária
    schedule.every().day.at(HORARIO_EXECUCAO).do(executar_worker_com_protecao)

    # Log de próxima execução
    proxima = schedule.next_run()
    logger.info(f"Próxima execução agendada: {proxima}")

    # Loop principal — checa a cada minuto se é hora de executar
    try:
        while True:
            schedule.run_pending()
            time.sleep(CHECK_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        logger.info("Serviço interrompido manualmente (Ctrl+C)")
    except Exception as e:
        logger.error(f"ERRO FATAL no loop do serviço: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()