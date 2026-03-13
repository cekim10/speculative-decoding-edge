import os

import torch

import util


class _ConfigMeta(type):
    _initialized = False

    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)
        return cls

    def __getattr__(cls, name):
        if name.startswith("_"):
            return super().__getattribute__(name)

        if not cls._initialized:
            cls._initialize()

        if name in cls.__dict__:
            return cls.__dict__[name]
        else:
            raise AttributeError(f"'{cls.__name__}' has no attribute '{name}'")

    def __setattr__(cls, name, value):
        super().__setattr__(name, value)

    def _initialize(cls):
        raise NotImplementedError("Subclasses must implement _initialize()")

    def _from_env(cls, key: str):
        value = os.getenv(key)

        if value is None or value == "null":
            raise ValueError(f"Environment variable '{key}' is not set")

        return value

    def reset(cls):
        cls._initialized = False
        cls._initialize()


class SpecEdgeClientConfig(metaclass=_ConfigMeta):
    """
    Configuration for the SpecEdge client

    Results and logs are stored in the directory
    "result_path/exp_name/process_name/seed"

    Attributes:
        optimization (int): Optimization level for the model
        result_path (str): Path to the directory where the results will be stored
        exp_name (str): Name of the experiment
        process_name (str): Name of the process

        seed (int): Seed for the random number generator

        draft_model (str): Path to the draft model
        device (torch.device): Device to run the model on
        dtype (torch.dtype): Data type to use for the model

        dataset (str): Name of the dataset

        max_n_beams (int): Maximum number of beams to generate
        max_beam_len (int): Maximum length of a beam
        max_branch_width (int): Maximum width of a branch
        max_budget (int): Maximum budget for the SpecExec algorithm

        proactive_type (str): Type of proactive draft
        proactive_max_n_beams (int): Maximum number of beams to generate proactively
        proactive_max_beam_len (int): Maximum length of a beam for proactive draft
        proactive_max_branch_width (int): Maximum width of a branch for proactive draft
        proactive_max_budget (int): Maximum budget for the proactive draft

        max_new_tokens (int): Maximum number of new tokens to generate
        max_request_num (int): Maximum number of requests to send

        host (str): Hostname of the server
        req_idx (int): Index of the request
    """

    @classmethod
    def _initialize(cls):
        # experiment configuration
        cls.result_path = cls._from_env("SPECEDGE_RESULT_PATH")
        cls.exp_name = cls._from_env("SPECEDGE_EXP_NAME")
        cls.process_name = cls._from_env("SPECEDGE_PROCESS_NAME")
        cls.seed = int(cls._from_env("SPECEDGE_SEED"))
        cls.optimization = int(cls._from_env("SPECEDGE_OPTIMIZATION"))
        cls.max_len = int(cls._from_env("SPECEDGE_MAX_LEN"))

        # model configuration
        cls.draft_model = cls._from_env("SPECEDGE_DRAFT_MODEL")
        cls.device = torch.device(cls._from_env("SPECEDGE_DEVICE"))
        cls.dtype = util.convert_dtype(cls._from_env("SPECEDGE_DTYPE"))
        cls.reasoning = cls._from_env("SPECEDGE_REASONING") == "True"

        # dataset configuration
        cls.dataset = cls._from_env("SPECEDGE_DATASET")

        # SpecExec configuration
        cls.max_n_beams = int(cls._from_env("SPECEDGE_MAX_N_BEAMS"))
        cls.max_beam_len = int(cls._from_env("SPECEDGE_MAX_BEAM_LEN"))
        cls.max_branch_width = int(cls._from_env("SPECEDGE_MAX_BRANCH_WIDTH"))
        cls.max_budget = int(cls._from_env("SPECEDGE_MAX_BUDGET"))

        # proactive draft configuration
        cls.proactive_type = cls._from_env("SPECEDGE_PROACTIVE_TYPE")
        cls.proactive_max_n_beams = int(cls._from_env("SPECEDGE_PROACTIVE_MAX_N_BEAMS"))
        cls.proactive_max_beam_len = int(
            cls._from_env("SPECEDGE_PROACTIVE_MAX_BEAM_LEN")
        )
        cls.proactive_max_branch_width = int(
            cls._from_env("SPECEDGE_PROACTIVE_MAX_BRANCH_WIDTH")
        )
        cls.proactive_max_budget = int(cls._from_env("SPECEDGE_PROACTIVE_MAX_BUDGET"))

        # token generation configuration
        cls.max_new_tokens = int(cls._from_env("SPECEDGE_MAX_NEW_TOKENS"))
        cls.max_request_num = int(cls._from_env("SPECEDGE_MAX_REQUEST_NUM"))
        cls.req_offset = int(cls._from_env("SPECEDGE_REQ_OFFSET"))
        cls.sample_req_cnt = int(cls._from_env("SPECEDGE_SAMPLE_REQ_CNT"))

        # server configuration
        cls.host = cls._from_env("SPECEDGE_HOST")
        cls.client_idx = int(cls._from_env("SPECEDGE_CLIENT_IDX"))

        # DASD configuration (optional)
        cls.mode = os.getenv("SPECEDGE_MODE", "specedge")
        cls.dasd_enable_async = os.getenv("SPECEDGE_DASD_ENABLE_ASYNC", "False") == "True"
        cls.dasd_start_window = int(os.getenv("SPECEDGE_DASD_START_WINDOW", "4"))
        cls.dasd_w_min = int(os.getenv("SPECEDGE_DASD_W_MIN", "4"))
        cls.dasd_w_max = int(os.getenv("SPECEDGE_DASD_W_MAX", str(cls.max_budget)))
        cls.dasd_alpha = int(os.getenv("SPECEDGE_DASD_ALPHA", "1"))
        cls.dasd_beta = float(os.getenv("SPECEDGE_DASD_BETA", "0.5"))
        cls.dasd_max_inflight_bundles = int(
            os.getenv("SPECEDGE_DASD_MAX_INFLIGHT_BUNDLES", "4")
        )
        cls.dasd_max_spec_buffer_tokens = int(
            os.getenv("SPECEDGE_DASD_MAX_SPEC_BUFFER_TOKENS", "256")
        )
        cls.dasd_abort_after_failures = int(
            os.getenv("SPECEDGE_DASD_ABORT_AFTER_FAILURES", "4")
        )
        cls.dasd_rollback_avoid_failed_token = (
            os.getenv("SPECEDGE_DASD_ROLLBACK_AVOID_FAILED_TOKEN", "False") == "True"
        )
        cls.dasd_adaptive_credit_enabled = (
            os.getenv("SPECEDGE_DASD_ADAPTIVE_CREDIT_ENABLED", "False") == "True"
        )
        cls.dasd_adaptive_window_enabled = (
            os.getenv("SPECEDGE_DASD_ADAPTIVE_WINDOW_ENABLED", "False") == "True"
        )
        cls.dasd_adaptive_tree_budget_enabled = (
            os.getenv("SPECEDGE_DASD_ADAPTIVE_TREE_BUDGET_ENABLED", "False") == "True"
        )
        cls.dasd_credit_min = int(os.getenv("SPECEDGE_DASD_CREDIT_MIN", "0"))
        cls.dasd_credit_max = int(
            os.getenv("SPECEDGE_DASD_CREDIT_MAX", str(max(cls.max_budget, cls.dasd_w_max)))
        )
        cls.dasd_credit_init = int(
            os.getenv("SPECEDGE_DASD_CREDIT_INIT", str(cls.dasd_start_window))
        )
        cls.dasd_rejection_penalty = int(
            os.getenv("SPECEDGE_DASD_REJECTION_PENALTY", "1")
        )
        cls.dasd_success_bonus = int(os.getenv("SPECEDGE_DASD_SUCCESS_BONUS", "1"))
        # Adaptive-window bounds. If these are not explicitly provided,
        # fall back to the legacy DASD window bounds for backward compatibility.
        cls.dasd_min_window = int(
            os.getenv("SPECEDGE_DASD_MIN_WINDOW", str(cls.dasd_w_min))
        )
        cls.dasd_max_window = int(
            os.getenv("SPECEDGE_DASD_MAX_WINDOW", str(cls.dasd_w_max))
        )
        cls.dasd_min_tree_depth = int(os.getenv("SPECEDGE_DASD_MIN_TREE_DEPTH", "1"))
        cls.dasd_max_tree_depth = int(
            os.getenv("SPECEDGE_DASD_MAX_TREE_DEPTH", str(cls.max_beam_len))
        )
        cls.dasd_min_leaf_budget = int(
            os.getenv("SPECEDGE_DASD_MIN_LEAF_BUDGET", "4")
        )
        cls.dasd_max_leaf_budget = int(
            os.getenv("SPECEDGE_DASD_MAX_LEAF_BUDGET", str(cls.max_budget))
        )
        cls.dasd_failure_cache_enabled = (
            os.getenv("SPECEDGE_DASD_FAILURE_CACHE_ENABLED", "False") == "True"
        )
        cls.dasd_failure_cache_cooldown = int(
            os.getenv("SPECEDGE_DASD_FAILURE_CACHE_COOLDOWN", "3")
        )
        cls.dasd_failure_cache_max_tokens_per_prefix = int(
            os.getenv("SPECEDGE_DASD_FAILURE_CACHE_MAX_TOKENS_PER_PREFIX", "4")
        )
        cls.dasd_recovery_mode_enabled = (
            os.getenv("SPECEDGE_DASD_RECOVERY_MODE_ENABLED", "False") == "True"
        )
        cls.dasd_recovery_full_rejection_threshold = int(
            os.getenv("SPECEDGE_DASD_RECOVERY_FULL_REJECTION_THRESHOLD", "3")
        )
        cls.dasd_recovery_same_base_retry_threshold = int(
            os.getenv("SPECEDGE_DASD_RECOVERY_SAME_BASE_RETRY_THRESHOLD", "3")
        )
        cls.dasd_recovery_mode_rounds = int(
            os.getenv("SPECEDGE_DASD_RECOVERY_MODE_ROUNDS", "2")
        )
        cls.dasd_recovery_forced_w = int(
            os.getenv("SPECEDGE_DASD_RECOVERY_FORCED_W", "1")
        )
        cls.dasd_recovery_forced_tree_depth = int(
            os.getenv("SPECEDGE_DASD_RECOVERY_FORCED_TREE_DEPTH", "1")
        )
        cls.dasd_recovery_forced_leaf_budget = int(
            os.getenv("SPECEDGE_DASD_RECOVERY_FORCED_LEAF_BUDGET", "2")
        )
        cls.dasd_recovery_forced_commit_enabled = (
            os.getenv("SPECEDGE_DASD_RECOVERY_FORCED_COMMIT_ENABLED", "True") == "True"
        )
        cls.dasd_recovery_forced_commit_same_base_retry_threshold = int(
            os.getenv(
                "SPECEDGE_DASD_RECOVERY_FORCED_COMMIT_SAME_BASE_RETRY_THRESHOLD", "3"
            )
        )
        cls.dasd_recovery_forced_commit_full_rejection_threshold = int(
            os.getenv(
                "SPECEDGE_DASD_RECOVERY_FORCED_COMMIT_FULL_REJECTION_THRESHOLD", "2"
            )
        )
        cls.dasd_recovery_forced_commit_max_per_base = int(
            os.getenv("SPECEDGE_DASD_RECOVERY_FORCED_COMMIT_MAX_PER_BASE", "1")
        )
        cls.dasd_recovery_fallback_decode_steps = int(
            os.getenv("SPECEDGE_DASD_RECOVERY_FALLBACK_DECODE_STEPS", "1")
        )
        cls.dasd_debug = os.getenv("SPECEDGE_DASD_DEBUG", "False") == "True"

        cls._initialized = True


class SpecEdgeBatchClientConfig(metaclass=_ConfigMeta):
    @classmethod
    def _initialize(cls):
        # experiment configuration
        cls.result_path = cls._from_env("SPECEDGE_RESULT_PATH")
        cls.exp_name = cls._from_env("SPECEDGE_EXP_NAME")
        cls.process_name = cls._from_env("SPECEDGE_PROCESS_NAME")
        cls.seed = int(cls._from_env("SPECEDGE_SEED"))
        cls.max_len = int(cls._from_env("SPECEDGE_MAX_LEN"))

        # model configuration
        cls.draft_model = cls._from_env("SPECEDGE_DRAFT_MODEL")
        cls.device = torch.device(cls._from_env("SPECEDGE_CLIENT_DEVICE"))
        cls.dtype = util.convert_dtype(cls._from_env("SPECEDGE_DTYPE"))

        # dataset configuration
        cls.dataset = cls._from_env("SPECEDGE_DATASET")

        # SpecExec configuration
        cls.max_n_beams = int(cls._from_env("SPECEDGE_MAX_N_BEAMS"))
        cls.max_beam_len = int(cls._from_env("SPECEDGE_MAX_BEAM_LEN"))
        cls.max_branch_width = int(cls._from_env("SPECEDGE_MAX_BRANCH_WIDTH"))
        cls.max_budget = int(cls._from_env("SPECEDGE_MAX_BUDGET"))

        # token generation configuration
        cls.max_batch_size = int(cls._from_env("SPECEDGE_MAX_BATCH_SIZE"))
        cls.max_new_tokens = int(cls._from_env("SPECEDGE_MAX_NEW_TOKENS"))
        cls.max_request_num = int(cls._from_env("SPECEDGE_MAX_REQUEST_NUM"))

        # server configuration
        cls.host = cls._from_env("SPECEDGE_HOST")
        cls.req_idx = int(cls._from_env("SPECEDGE_REQ_IDX"))
        cls.sample_req_cnt = int(cls._from_env("SPECEDGE_SAMPLE_REQ_CNT"))

        cls._initialized = True


class SpecEdgeServerConfig(metaclass=_ConfigMeta):
    @classmethod
    def _initialize(cls):
        cls.result_path = cls._from_env("SPECEDGE_RESULT_PATH")
        cls.exp_name = cls._from_env("SPECEDGE_EXP_NAME")
        cls.process_name = cls._from_env("SPECEDGE_PROCESS_NAME")
        cls.seed = int(cls._from_env("SPECEDGE_SEED"))
        cls.optimization = int(cls._from_env("SPECEDGE_OPTIMIZATION"))
        cls.max_len = int(cls._from_env("SPECEDGE_MAX_LEN"))

        # model configuration
        cls.target_model = cls._from_env("SPECEDGE_TARGET_MODEL")
        cls.device = torch.device(cls._from_env("SPECEDGE_DEVICE"))
        cls.dtype = util.convert_dtype(cls._from_env("SPECEDGE_DTYPE"))
        cls.temperature = float(cls._from_env("SPECEDGE_TEMPERATURE"))

        # engine configuration
        cls.max_n_beams = int(cls._from_env("SPECEDGE_MAX_N_BEAMS"))

        cls._initialized = True


class SpecEdgeBatchServerConfig(metaclass=_ConfigMeta):
    @classmethod
    def _initialize(cls):
        cls.result_path = cls._from_env("SPECEDGE_RESULT_PATH")
        cls.exp_name = cls._from_env("SPECEDGE_EXP_NAME")
        cls.process_name = cls._from_env("SPECEDGE_PROCESS_NAME")
        cls.seed = int(cls._from_env("SPECEDGE_SEED"))
        cls.max_len = int(cls._from_env("SPECEDGE_MAX_LEN"))
        cls.batch_type = cls._from_env("SPECEDGE_BATCH_TYPE")
        cls.dataset = cls._from_env("SPECEDGE_DATASET")
        cls.sample_req_cnt = int(cls._from_env("SPECEDGE_SAMPLE_REQ_CNT"))
        cls.req_offset = int(cls._from_env("SPECEDGE_REQ_OFFSET"))

        # model configuration
        cls.target_model = cls._from_env("SPECEDGE_TARGET_MODEL")
        cls.device = torch.device(cls._from_env("SPECEDGE_SERVER_DEVICE"))
        cls.dtype = util.convert_dtype(cls._from_env("SPECEDGE_DTYPE"))
        cls.temperature = float(cls._from_env("SPECEDGE_TEMPERATURE"))

        # engine configuration
        cls.max_batch_size = int(cls._from_env("SPECEDGE_MAX_BATCH_SIZE"))
        cls.max_n_beams = int(cls._from_env("SPECEDGE_MAX_N_BEAMS"))
        cls.max_budget = int(cls._from_env("SPECEDGE_MAX_BUDGET"))
        cls.num_clients = int(cls._from_env("SPECEDGE_NUM_CLIENTS"))
        cls.cache_prefill = cls._from_env("SPECEDGE_CACHE_PREFILL") == "True"

        # DASD mode configuration (optional; defaults preserve baseline behavior)
        cls.mode = os.getenv("SPECEDGE_MODE", "specedge")
        cls.dasd_enable_async = os.getenv("SPECEDGE_DASD_ENABLE_ASYNC", "False") == "True"
        cls.dasd_w_min = int(os.getenv("SPECEDGE_DASD_W_MIN", "4"))
        cls.dasd_w_max = int(os.getenv("SPECEDGE_DASD_W_MAX", "32"))
        cls.dasd_alpha = int(os.getenv("SPECEDGE_DASD_ALPHA", "1"))
        cls.dasd_beta = float(os.getenv("SPECEDGE_DASD_BETA", "0.5"))
        cls.dasd_max_inflight_bundles = int(
            os.getenv("SPECEDGE_DASD_MAX_INFLIGHT_BUNDLES", "8")
        )
        cls.dasd_max_spec_buffer_tokens = int(
            os.getenv("SPECEDGE_DASD_MAX_SPEC_BUFFER_TOKENS", "512")
        )
        cls.dasd_global_budget_c_total = int(
            os.getenv("SPECEDGE_DASD_GLOBAL_BUDGET_C_TOTAL", "0")
        )
        cls.dasd_ema_decay = float(os.getenv("SPECEDGE_DASD_EMA_DECAY", "0.9"))
        cls.dasd_debug = os.getenv("SPECEDGE_DASD_DEBUG", "False") == "True"

        cls._initialized = True


class AutoregressiveBatchConfig(metaclass=_ConfigMeta):
    @classmethod
    def _initialize(cls):
        cls.result_path = cls._from_env("SPECEDGE_RESULT_PATH")
        cls.exp_name = cls._from_env("SPECEDGE_EXP_NAME")
        cls.process_name = cls._from_env("SPECEDGE_PROCESS_NAME")
        cls.seed = int(cls._from_env("SPECEDGE_SEED"))

        cls.model = cls._from_env("SPECEDGE_MODEL")
        cls.device = torch.device(cls._from_env("SPECEDGE_DEVICE"))
        cls.dtype = util.convert_dtype(cls._from_env("SPECEDGE_DTYPE"))
        cls.temperature = float(cls._from_env("SPECEDGE_TEMPERATURE"))

        cls.dataset = cls._from_env("SPECEDGE_DATASET")

        cls.max_len = int(cls._from_env("SPECEDGE_MAX_LEN"))
        cls.max_new_tokens = int(cls._from_env("SPECEDGE_MAX_NEW_TOKENS"))
        cls.max_request_num = int(cls._from_env("SPECEDGE_MAX_REQUEST_NUM"))
        cls.batch_size = int(cls._from_env("SPECEDGE_BATCH_SIZE"))
        cls.sample_req_cnt = int(cls._from_env("SPECEDGE_SAMPLE_REQ_CNT"))

        cls._initialized = True
