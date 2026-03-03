import argparse
import asyncio
import os
import signal
from pathlib import Path

import grpc.aio
import yaml

import log
import util
from config import SpecEdgeBatchServerConfig as config
from specedge_grpc import specedge_pb2_grpc
from strategy.server_verify.specexec.grpc import SpecExecBatchServer

shutdown_event = None


def signal_handler(signum, frame):
    """Handle shutdown signals (SIGINT, SIGTERM)"""
    if shutdown_event:
        shutdown_event.set()


async def serve():
    global shutdown_event

    shutdown_event = asyncio.Event()
    controller = SpecExecBatchServer(shutdown_event=shutdown_event)

    server = grpc.aio.server()
    specedge_pb2_grpc.add_SpecEdgeServiceServicer_to_server(controller, server)
    server.add_insecure_port("[::]:8000")

    try:
        await server.start()
        await shutdown_event.wait()

        await server.stop(grace=2.0)
        await controller.cleanup()

    except asyncio.CancelledError:
        await server.stop(0)

    except Exception as e:
        await server.stop(0)
        raise


def _load_config(config_file: Path):
    with open(config_file, "r") as f:
        config_yaml = yaml.safe_load(f)

    result_path = config_yaml["base"]["result_path"]
    exp_name = config_yaml["base"]["exp_name"]
    process_name = "server"
    seed = config_yaml["base"]["seed"]
    max_len = config_yaml["base"]["max_len"]
    batch_type = config_yaml["server"]["batch_type"]
    dataset = config_yaml["client"]["dataset"]
    sample_req_cnt = config_yaml["client"]["sample_req_cnt"]
    req_offset = config_yaml["client"]["req_offset"]

    target_model = config_yaml["server"]["target_model"]
    device = config_yaml["server"]["device"]
    dtype = config_yaml["base"]["dtype"]
    temperature = config_yaml["server"]["temperature"]

    max_batch_size = config_yaml["server"]["max_batch_size"]
    max_n_beams = config_yaml["client"]["max_n_beams"]
    max_budget = config_yaml["client"]["max_budget"]
    num_clients = config_yaml["server"]["num_clients"]
    cache_prefill = config_yaml["server"]["cache_prefill"]
    mode = config_yaml.get("mode", "specedge")
    dasd_cfg = config_yaml.get("dasd", {})

    os.environ["SPECEDGE_RESULT_PATH"] = result_path
    os.environ["SPECEDGE_EXP_NAME"] = exp_name
    os.environ["SPECEDGE_PROCESS_NAME"] = process_name
    os.environ["SPECEDGE_SEED"] = str(seed)
    os.environ["SPECEDGE_MAX_LEN"] = str(max_len)
    os.environ["SPECEDGE_BATCH_TYPE"] = batch_type
    os.environ["SPECEDGE_DATASET"] = dataset
    os.environ["SPECEDGE_SAMPLE_REQ_CNT"] = str(sample_req_cnt)
    os.environ["SPECEDGE_REQ_OFFSET"] = str(req_offset)

    os.environ["SPECEDGE_TARGET_MODEL"] = target_model
    os.environ["SPECEDGE_SERVER_DEVICE"] = device
    os.environ["SPECEDGE_DTYPE"] = dtype
    os.environ["SPECEDGE_TEMPERATURE"] = str(temperature)

    os.environ["SPECEDGE_MAX_BATCH_SIZE"] = str(max_batch_size)
    os.environ["SPECEDGE_MAX_N_BEAMS"] = str(max_n_beams)
    os.environ["SPECEDGE_MAX_BUDGET"] = str(max_budget)

    os.environ["SPECEDGE_NUM_CLIENTS"] = str(num_clients)
    os.environ["SPECEDGE_CACHE_PREFILL"] = str(cache_prefill)
    os.environ["SPECEDGE_MODE"] = str(mode)
    os.environ["SPECEDGE_DASD_ENABLE_ASYNC"] = str(
        dasd_cfg.get("enable_async", False)
    )
    os.environ["SPECEDGE_DASD_W_MIN"] = str(dasd_cfg.get("W_min", 4))
    os.environ["SPECEDGE_DASD_W_MAX"] = str(dasd_cfg.get("W_max", max_budget))
    os.environ["SPECEDGE_DASD_ALPHA"] = str(dasd_cfg.get("alpha", 1))
    os.environ["SPECEDGE_DASD_BETA"] = str(dasd_cfg.get("beta", 0.5))
    os.environ["SPECEDGE_DASD_MAX_INFLIGHT_BUNDLES"] = str(
        dasd_cfg.get("max_inflight_bundles", 8)
    )
    os.environ["SPECEDGE_DASD_MAX_SPEC_BUFFER_TOKENS"] = str(
        dasd_cfg.get("max_spec_buffer_tokens", 512)
    )
    os.environ["SPECEDGE_DASD_GLOBAL_BUDGET_C_TOTAL"] = str(
        dasd_cfg.get("global_budget_C_total", 0)
    )
    os.environ["SPECEDGE_DASD_EMA_DECAY"] = str(dasd_cfg.get("ema_decay", 0.9))
    os.environ["SPECEDGE_DASD_DEBUG"] = str(dasd_cfg.get("debug", False))

    log_config = log.get_default_log_config(
        Path(config.result_path) / config.exp_name, "server"
    )
    log.configure_logging(log_config)
    log.log_unexpected_exception()

    logger = log.get_logger()

    logger.debug("result_path: %s", result_path)
    logger.debug("exp_name: %s", exp_name)
    logger.debug("process_name: %s", process_name)
    logger.debug("seed: %s", seed)
    logger.debug("max_len: %s", max_len)
    logger.debug("target_model: %s", target_model)
    logger.debug("device: %s", device)
    logger.debug("dtype: %s", dtype)
    logger.debug("temperature: %s", temperature)
    logger.debug("mode: %s", mode)
    logger.debug("max_batch_size: %s", max_batch_size)
    logger.debug("max_n_beams: %s", max_n_beams)
    logger.debug("max_budget: %s", max_budget)
    logger.debug("dasd_cfg: %s", dasd_cfg)
    logger.info("Config loaded successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()

    _load_config(Path(args.config))

    util.set_seed(config.seed)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger = log.get_logger()

    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        # Signal handler will take care of graceful shutdown
        pass
    except Exception as e:
        logger.exception("Fatal error: %s", e)
    finally:
        import logging

        logging.shutdown()
