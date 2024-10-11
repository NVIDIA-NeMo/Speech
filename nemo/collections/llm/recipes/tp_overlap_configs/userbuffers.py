from dataclasses import dataclass


@dataclass
class TPOverlapCfg:
    pass


@dataclass
class PipelineOverlapCfg(TPOverlapCfg):
    num_sm: int
    cga_size: int
    num_splits: int
    set_sm_margin: bool
    fp8_buf: bool = (False,)
    method: str = 'pipeline'


@dataclass
class RingExchangeOverlapCfg(TPOverlapCfg):
    aggregate: bool = False
    method: str = 'ring_exchange'
    num_sm: int = 1
    set_sm_margin: bool = False


@dataclass
class BulkOverlapCfg(TPOverlapCfg):
    num_sm: int
    cga_size: int
    set_sm_margin: bool
    method: str = 'bulk'


@dataclass
class TransformerLayerTPOverlapCfg:
    qkv_dgrad: TPOverlapCfg
    qkv_wgrad: TPOverlapCfg
    fc1_dgrad: TPOverlapCfg
    fc1_wgrad: TPOverlapCfg
    qkv_fprop: TPOverlapCfg
    proj_dgrad: TPOverlapCfg
    fc1_fprop: TPOverlapCfg
    fc2_dgrad: TPOverlapCfg
    proj_fprop: TPOverlapCfg
    fc2_fprop: TPOverlapCfg


# TODO: Add more configs and create a getter function for expose a single api
# Model configs: H100/70B/TP8/MBS1/SeqLen8K
userbuffers_bf16_h100_h8192_tp4_mbs1_seqlen8192 = TransformerLayerTPOverlapCfg(
    qkv_dgrad=BulkOverlapCfg(num_sm=4, cga_size=2, set_sm_margin=False),
    qkv_wgrad=BulkOverlapCfg(num_sm=24, cga_size=2, set_sm_margin=False),
    fc1_dgrad=BulkOverlapCfg(num_sm=2, cga_size=2, set_sm_margin=False),
    fc1_wgrad=BulkOverlapCfg(num_sm=4, cga_size=2, set_sm_margin=False),
    qkv_fprop=RingExchangeOverlapCfg(aggregate=False),
    proj_dgrad=RingExchangeOverlapCfg(aggregate=False),
    fc1_fprop=RingExchangeOverlapCfg(aggregate=False),
    fc2_dgrad=RingExchangeOverlapCfg(aggregate=False),
    proj_fprop=PipelineOverlapCfg(num_sm=24, cga_size=2, num_splits=4, set_sm_margin=True),
    fc2_fprop=PipelineOverlapCfg(num_sm=16, cga_size=2, num_splits=4, set_sm_margin=True),
)

userbuffers_fp8_h100_h8192_tp4_mbs1_seqlen8192 = TransformerLayerTPOverlapCfg(
    qkv_dgrad=BulkOverlapCfg(num_sm=4, cga_size=2, set_sm_margin=False),
    qkv_wgrad=BulkOverlapCfg(num_sm=24, cga_size=2, set_sm_margin=False),
    fc1_dgrad=BulkOverlapCfg(num_sm=2, cga_size=2, set_sm_margin=False),
    fc1_wgrad=BulkOverlapCfg(num_sm=4, cga_size=2, set_sm_margin=False),
    qkv_fprop=RingExchangeOverlapCfg(aggregate=False),
    proj_dgrad=RingExchangeOverlapCfg(aggregate=False),
    fc1_fprop=RingExchangeOverlapCfg(aggregate=False),
    fc2_dgrad=RingExchangeOverlapCfg(aggregate=False),
    proj_fprop=PipelineOverlapCfg(num_sm=24, cga_size=2, num_splits=4, set_sm_margin=True, fp8_buf=True),
    fc2_fprop=PipelineOverlapCfg(num_sm=16, cga_size=2, num_splits=4, set_sm_margin=True, fp8_buf=True),
)

# llama3.1 405b
userbuffers_bf16_h100_h16384_tp8_cp2_mbs1_seqlen8192 = TransformerLayerTPOverlapCfg(
    qkv_dgrad=BulkOverlapCfg(num_sm=2, cga_size=2, set_sm_margin=False),
    qkv_wgrad=BulkOverlapCfg(num_sm=24, cga_size=2, set_sm_margin=False),
    fc1_dgrad=BulkOverlapCfg(num_sm=2, cga_size=2, set_sm_margin=False),
    fc1_wgrad=BulkOverlapCfg(num_sm=2, cga_size=2, set_sm_margin=False),
    qkv_fprop=RingExchangeOverlapCfg(aggregate=True),
    proj_dgrad=RingExchangeOverlapCfg(aggregate=True),
    fc1_fprop=RingExchangeOverlapCfg(aggregate=True),
    fc2_dgrad=RingExchangeOverlapCfg(aggregate=True),
    proj_fprop=PipelineOverlapCfg(num_sm=24, cga_size=2, num_splits=4, set_sm_margin=True),
    fc2_fprop=PipelineOverlapCfg(num_sm=8, cga_size=2, num_splits=4, set_sm_margin=True),
)

userbuffers_fp8_h100_h16384_tp8_cp2_mbs1_seqlen8192 = TransformerLayerTPOverlapCfg(
    qkv_dgrad=BulkOverlapCfg(num_sm=2, cga_size=2, set_sm_margin=False),
    qkv_wgrad=BulkOverlapCfg(num_sm=24, cga_size=2, set_sm_margin=False),
    fc1_dgrad=BulkOverlapCfg(num_sm=2, cga_size=2, set_sm_margin=False),
    fc1_wgrad=BulkOverlapCfg(num_sm=2, cga_size=2, set_sm_margin=False),
    qkv_fprop=RingExchangeOverlapCfg(aggregate=True),
    proj_dgrad=RingExchangeOverlapCfg(aggregate=True),
    fc1_fprop=RingExchangeOverlapCfg(aggregate=True),
    fc2_dgrad=RingExchangeOverlapCfg(aggregate=True),
    proj_fprop=PipelineOverlapCfg(num_sm=24, cga_size=2, num_splits=4, set_sm_margin=True, fp8_buf=True),
    fc2_fprop=PipelineOverlapCfg(num_sm=8, cga_size=2, num_splits=4, set_sm_margin=True, fp8_buf=True),
)

# GPT3 20b
userbuffers_bf16_h100_h6144_tp2_mbs2_seqlen2048 = TransformerLayerTPOverlapCfg(
    qkv_dgrad=BulkOverlapCfg(num_sm=8, cga_size=2, set_sm_margin=False),
    qkv_wgrad=BulkOverlapCfg(num_sm=16, cga_size=2, set_sm_margin=False),
    fc1_dgrad=BulkOverlapCfg(num_sm=4, cga_size=2, set_sm_margin=False),
    fc1_wgrad=BulkOverlapCfg(num_sm=16, cga_size=2, set_sm_margin=False),
    qkv_fprop=RingExchangeOverlapCfg(aggregate=False),
    proj_dgrad=RingExchangeOverlapCfg(aggregate=False),
    fc1_fprop=RingExchangeOverlapCfg(aggregate=False),
    fc2_dgrad=RingExchangeOverlapCfg(aggregate=False),
    proj_fprop=PipelineOverlapCfg(num_sm=16, cga_size=2, num_splits=4, set_sm_margin=True),
    fc2_fprop=PipelineOverlapCfg(num_sm=24, cga_size=2, num_splits=4, set_sm_margin=True),
)

userbuffers_fp8_h100_h6144_tp2_mbs2_seqlen2048 = TransformerLayerTPOverlapCfg(
    qkv_dgrad=BulkOverlapCfg(num_sm=8, cga_size=2, set_sm_margin=False),
    qkv_wgrad=BulkOverlapCfg(num_sm=16, cga_size=2, set_sm_margin=False),
    fc1_dgrad=BulkOverlapCfg(num_sm=4, cga_size=2, set_sm_margin=False),
    fc1_wgrad=BulkOverlapCfg(num_sm=16, cga_size=2, set_sm_margin=False),
    qkv_fprop=RingExchangeOverlapCfg(aggregate=False),
    proj_dgrad=RingExchangeOverlapCfg(aggregate=False),
    fc1_fprop=RingExchangeOverlapCfg(aggregate=False),
    fc2_dgrad=RingExchangeOverlapCfg(aggregate=False),
    proj_fprop=PipelineOverlapCfg(num_sm=16, cga_size=2, num_splits=4, set_sm_margin=True, fp8_buf=True),
    fc2_fprop=PipelineOverlapCfg(num_sm=24, cga_size=2, num_splits=4, set_sm_margin=True, fp8_buf=True),
)

# GPT3 175b
userbuffers_bf16_h100_h12288_tp4_mbs1_seqlen2048 = TransformerLayerTPOverlapCfg(
    qkv_dgrad=BulkOverlapCfg(num_sm=4, cga_size=2, set_sm_margin=False),
    qkv_wgrad=BulkOverlapCfg(num_sm=4, cga_size=2, set_sm_margin=False),
    fc1_dgrad=BulkOverlapCfg(num_sm=2, cga_size=2, set_sm_margin=False),
    fc1_wgrad=BulkOverlapCfg(num_sm=4, cga_size=2, set_sm_margin=False),
    qkv_fprop=RingExchangeOverlapCfg(aggregate=False),
    proj_dgrad=RingExchangeOverlapCfg(aggregate=False),
    fc1_fprop=RingExchangeOverlapCfg(aggregate=False),
    fc2_dgrad=RingExchangeOverlapCfg(aggregate=False),
    proj_fprop=PipelineOverlapCfg(num_sm=24, cga_size=2, num_splits=4, set_sm_margin=True),
    fc2_fprop=RingExchangeOverlapCfg(num_sm=1, set_sm_margin=True),
)

userbuffers_fp8_h100_h12288_tp4_mbs1_seqlen2048 = TransformerLayerTPOverlapCfg(
    qkv_dgrad=BulkOverlapCfg(num_sm=4, cga_size=2, set_sm_margin=False),
    qkv_wgrad=BulkOverlapCfg(num_sm=4, cga_size=2, set_sm_margin=False),
    fc1_dgrad=BulkOverlapCfg(num_sm=2, cga_size=2, set_sm_margin=False),
    fc1_wgrad=BulkOverlapCfg(num_sm=4, cga_size=2, set_sm_margin=False),
    qkv_fprop=RingExchangeOverlapCfg(aggregate=False),
    proj_dgrad=RingExchangeOverlapCfg(aggregate=False),
    fc1_fprop=RingExchangeOverlapCfg(aggregate=False),
    fc2_dgrad=RingExchangeOverlapCfg(aggregate=False),
    proj_fprop=PipelineOverlapCfg(num_sm=24, cga_size=2, num_splits=4, set_sm_margin=True, fp8_buf=True),
    fc2_fprop=RingExchangeOverlapCfg(num_sm=1, set_sm_margin=True),
)
