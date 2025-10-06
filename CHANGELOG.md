# Changelog

<!-- Next changelog -->
## NVIDIA Neural Modules 2.4.1

### Detailed Changelogs:

#### Uncategorized:

<details><summary>Changelog</summary>

- Update package_info.py by @ko3n1g :: PR: #14400
- Patch to address issue 14392 by @youngeunkwon0405 :: PR: #14398
- Cherry pick `Fix callbacks in DSV3 script (14350)` into `r2.4.0` by @chtruong814 :: PR: #14370
- Cherry pick `Change Llama Embedding Tutorial to use SFT by default (14231)` into `r2.4.0` by @chtruong814 :: PR: #14303
- Cherrypick `calculate_per_token_loss requirement for context parallel` (#14065) (#14282) into `r2.4.0` by @chtruong814 :: PR: #14448
- Pin nvidia-lm-eval to 25.6.1 by @chtruong814 :: PR: #14470

</details>

## NVIDIA Neural Modules 2.3.3

- This release addresses known security issues. For the latest NVIDIA Vulnerability Disclosure Information visit <https://www.nvidia.com/en-us/security/>, for acknowledgement please reach out to the NVIDIA PSIRT team at <PSIRT@nvidia.com>
- Pin nvidia-lm-eval to 25.5

## NVIDIA Neural Modules 2.4.0

### Highlights

- Collections:
  - Speech
    - Batched beam search for transducers (RNN-T and TDT)
      - RNNT/TDT buffered/streaming inference \+ batched decoding support in cache-aware
      - add support for CTC batched beam search with GPU-LM
      - Key fixes
        - Punctuation Marks in Timestamps
        - Fix timestamps when cuda graphs enabled
        - Fix masking of \<pad\> tokens in AED inference
        - TDT streaming inference fix
  - LLM
    - Qwen 3 235B-A22B Perf Optimized
    - DeepSeek V3 Perf Optimized
    - Gemma3 support from Google
    - Embedding and Reranker models
  - MM
    - Llama 4
    - AVLM
- Training performance (speed)
  - NVL sharp \+ IB sharp for DP/FSDP-communications on H100 and B200
  - MXFP8 with TP communication overlap
  - MXFP8 with reduced memory allocation
  - FP8 sub-channel recipe (128x128 for weight and 1x128 for activation)
  - cudnn fused attention for MLA (both Hopper and Blackwell)
  - Advanced custom asymmetric pipelining (for MTP, loss func, and embd)
  - BF16 optimizer for model memory saving
  - CUDA graph fix for fine-tuning benchmarks
  - CUDA graph support for LLAMA4

### Detailed Changelogs

#### ASR

<details><summary>Changelog</summary>

- ci: Fix ASR container by @ko3n1g :: PR: #13288
- Set L2_Segmentation_Tool_Parallel_ctc_segmentation test to be optional by @chtruong814 :: PR: #13296
- Revert "WebDataset URL refactoring" by @ko3n1g :: PR: #13421
- Update flagged docs links by @erastorgueva-nv :: PR: #13391
- [Docs] Fix incorrectly formatted reference tags by @erastorgueva-nv :: PR: #13445
- Update CP by @pablo-garay :: PR: #13532
- Tdt buffered inference fix by @hainan-xv :: PR: #13500
- Fix transcribe when nbest hypotheses are returned by @lilithgrigoryan :: PR: #13540
- Set ASR test to be optional by @chtruong814 :: PR: #13633
- Enabling chunked inference for AED models in asr_evaluator by @melllinia :: PR: #13674
- Ko3n1g/chore/asr only by @ko3n1g :: PR: #13704
- decompressing joblib file before checking it by @Ssofja :: PR: #13732
- Revert "decompressing joblib file before checking it (#13732)" by @chtruong814 :: PR: #13791
- Punctuation Marks in Timestamps by @monica-sekoyan :: PR: #13353
- AIStore with Webdataset by @monica-sekoyan :: PR: #13604
- Update to add default for dataclass variables by @nithinraok :: PR: #13814
- This PR addresses to known security issues by @Ssofja :: PR: #13804
- remove model_stride var by @nithinraok :: PR: #13867
- add CTC batched beam search by @lilithgrigoryan :: PR: #13337
- Clean up streaming ASR script and tests by @artbataev :: PR: #13894
- add NGPU-LM fusion during CTC greedy by @lilithgrigoryan :: PR: #13917

</details>

#### TTS

<details><summary>Changelog</summary>

- Revert "WebDataset URL refactoring" by @ko3n1g :: PR: #13421
- Update flagged docs links by @erastorgueva-nv :: PR: #13391
- [Docs] Fix incorrectly formatted reference tags by @erastorgueva-nv :: PR: #13445
- Update CP by @pablo-garay :: PR: #13532
- fix: vpp stage refactoring to match mcore  by @ZhiyuLi-Nvidia :: PR: #13673
- AIStore with Webdataset by @monica-sekoyan :: PR: #13604

</details>

#### NLP / NMT

<details><summary>Changelog</summary>

- Migrate Hyena to Megatron inference_context. by @cspades :: PR: #13436
- Update CP by @pablo-garay :: PR: #13532
- fix broken links by @dimapihtar :: PR: #13544
- Add nlp import checks by @thomasdhc :: PR: #13563
- PTQ model support, quant_cfg, and documentation updates by @janekl :: PR: #13519
- feat - GPTSFTChatDataset alignment with OpenAI Messages, compatibility with packed sequences by @soluwalana :: PR: #13367
- fix: vpp stage refactoring to match mcore  by @ZhiyuLi-Nvidia :: PR: #13673
- Fix resume with MegatronPretrainingBatchSampler by @ashors1 :: PR: #13565
- Punctuation Marks in Timestamps by @monica-sekoyan :: PR: #13353
- Revert `Adding more doc-strings to megatron_parallel.py #12767` by @ko3n1g :: PR: #13824
- reasoning model evaluation mmlu gpqa by @ruchaa-apte :: PR: #13880
- Remove unused DynamicRetrievalServer and Bert dataset loader classes by @dimapihtar :: PR: #14209
- Huvu/avlm qafix cherrypick from  by @huvunvidia :: PR: #14253

</details>

#### Export

<details><summary>Changelog</summary>

- Improve Nemo2Exporter for Models Using Custom Modelling Files on HF by @suiyoubi :: PR: #13400
- Adding more export tests by @oyilmaz-nvidia :: PR: #13410
- Add Warning to Export when output_path exists by @suiyoubi :: PR: #13465
- Move libsox-fmt-all from Dockerfile.ci.export_deploy to Dockerfile.ci by @chtruong814 :: PR: #13452
- ci: Remove trt-llm breakpoint by @ko3n1g :: PR: #13499
- Add Qwen2VL export_ckpt by @AtsunoriFujita :: PR: #13398
- Add MLlama export_ckpt by @AtsunoriFujita :: PR: #13346
- Update vLLMExporter to use vLLM V1 by @janekl :: PR: #13498
- Add vLLM Mixtral and TRT-LLM qnemo export tests (plus a couple of bugfixes) by @janekl :: PR: #13697
- Fix Qwen3 export + misc by @cuichenx :: PR: #13679
- Extra int cast for successful tracing during ONNX export by @janekl :: PR: #13782
- FP8 lora export by @cuichenx :: PR: #13748
- Add PEFT export check by @cuichenx :: PR: #13835
- Update llm api import_ckpt/export_ckpt docstring by @meatybobby :: PR: #13714
- Use modelopt export and disable dataset calibration for weight only PTQ by @jenchen13 :: PR: #13756

</details>

#### Bugfixes

<details><summary>Changelog</summary>

- [automodel] move liger kernel patching by @akoumpa :: PR: #13579

</details>

#### Uncategorized

<details><summary>Changelog</summary>

- build: various bumps by @ko3n1g :: PR: #13285
- ci: Fixes to selective triggering by @ko3n1g :: PR: #13287
- ci: Set timeout by @ko3n1g :: PR: #13294
- Set L2_NeMo_2_T5_Pretraining test as optional by @chtruong814 :: PR: #13282
- Add test environment approval step for CI by @chtruong814 :: PR: #13297
- update num nodes in deepseek v3 finetune recipe by @cuichenx :: PR: #13314
- ci: Increase cache pool by @ko3n1g :: PR: #13306
- Rename adam_with_cosine_annealing as adam since cosin LR is not setup by @ShriyaRishab :: PR: #13315
- ci: Update test queue bot to not assume a workflow is launched from a PR by @chtruong814 :: PR: #13318
- Fix TE pytorch attention doc link by @thomasdhc :: PR: #13327
- ci: Add all recent buildcaches to update-buildcache job by @ko3n1g :: PR: #13289
- Fix neva notebook by @yaoyu-33 :: PR: #13334
- Fix transformer offline for CI/CD llama4 tests by @yaoyu-33 :: PR: #13339
- [automodel] convert lm head to full tensor before passing to lce by @yuanzhedong :: PR: #13319
- ci: No dups in queue by @ko3n1g :: PR: #13352
- ci(hotfix): VLM CPU unit tests by @ko3n1g :: PR: #13348
- vLLM==0.8.5 update  by @janekl :: PR: #13350
- ci: Allow bypassing approval by @ko3n1g :: PR: #13365
- Avoid the need to specify optional attributes for lhotse/nemo reader functions by @pzelasko :: PR: #13307
- ci: Fix selective-triggering for non-PR events by @ko3n1g :: PR: #13374
- ci: Revert `no-concurrency-group-on-main` by @ko3n1g :: PR: #13375
- ci: Improve no-fail-fast mechanism by @ko3n1g :: PR: #13370
- 2d buckets estimation fix by @monica-sekoyan :: PR: #13377
- ci: Fix scheduled runs by @ko3n1g :: PR: #13378
- Ko3n1g/ci/fix nightly runs by @ko3n1g :: PR: #13382
- [automodel] fix none issue in dataset for qwen model by @yuanzhedong :: PR: #13311
- update table by @akoumpa :: PR: #13397
- Improve test coverage for audio modules by @anteju :: PR: #13333
- Disable failing maxine loss test by @anteju :: PR: #13361
- Ko3n1g/ci/no notification on cancel by @ko3n1g :: PR: #13403
- document fp8_recipe by @akoumpa :: PR: #13405
- Weekly bump main by @ko3n1g :: PR: #13408
- Handle boolean args for performance scripts and log received config by @guyueh1 :: PR: #13291
- [automodel] add FirstRankPerNode by @akoumpa :: PR: #13373
- tests: Disable flaky audio test by @ko3n1g :: PR: #13429
- ci: Disable flaky audio test by @ko3n1g :: PR: #13435
- Fix loss compute and reduction by @xrennvidia :: PR: #13295
- ci: Skip link check on github links by @chtruong814 :: PR: #13425
- Add NCCL cfg interface to perf scripts by @erhoo82 :: PR: #13407
- ci: Success only if `Run CICD` label attached by @ko3n1g :: PR: #13430
- ci: Add tests to selective triggering by @ko3n1g :: PR: #13404
- ci: Remove jq by @ko3n1g :: PR: #13440
- ci: Fix deps tree for tests by @ko3n1g :: PR: #13443
- Ko3n1g/ci/fix dependency tree by @ko3n1g :: PR: #13448
- Adding additional unit tests for the deploy module by @pthombre :: PR: #13411
- [Audio] fix a flaky test (and also make some tests run faster) by @racoiaws :: PR: #13439
- [automodel] ignore tail padding in TPS calculation by @akoumpa :: PR: #13329
- Ko3n1g/ci/selective triggering 3 by @ko3n1g :: PR: #13460
- ci: Disable broken neva tests by @ko3n1g :: PR: #13461
- fix speechlm data module by @stevehuang52 :: PR: #13362
- ci: Enter queue only with passing linting by @ko3n1g :: PR: #13462
- Adding tests for Schroedinger Bridge model by @nasretdinovr :: PR: #13401
- add more detailed description by @dimapihtar :: PR: #13464
- [Audio] tests for score-based and flow matching enhancement models by @racoiaws :: PR: #13406
- Use expandable cuda memory segmentation by @erhoo82 :: PR: #13418
- Fix llava tokenizer caused nan issue by @yaoyu-33 :: PR: #13466
- Remove cuda method from ModelPT by @erastorgueva-nv :: PR: #13394
- Fix BNR 2 unit test + input, case where input length was not specified by @nitin9252 :: PR: #13467
- ci: Do not run any tests if no match is found by @ko3n1g :: PR: #13479
- Ko3n1g/ci/selective triggering 4 by @ko3n1g :: PR: #13489
- Fix typo in the performance script by @youngeunkwon0405 :: PR: #13487
- ci: No runs on main by @ko3n1g :: PR: #13490
- ci: Upload on schedule by @ko3n1g :: PR: #13491
- ci: Run selective triggering on dockerfiles and dependencies by @ko3n1g :: PR: #13493
- [automodel] fallback FP8 + LCE -> FP8 + CE  by @akoumpa :: PR: #13349
- Update changelog for `r2.3.0` by @github-actions[bot] :: PR: #13501
- Update 2.3.0 changelog by @chtruong814 :: PR: #13504
- Enabling flash decode for float16 precision only by @pthombre :: PR: #13471
- Fix changelog formatting by @chtruong814 :: PR: #13505
- Updating the long context performance number for B200 by @youngeunkwon0405 :: PR: #13468
- ci: Add more files to filter by @ko3n1g :: PR: #13517
- Improve error message when HF checkpoint cannot be loaded by @ashors1 :: PR: #13513
- Add Resume_path to llama_nemotron models by @suiyoubi :: PR: #13515
- Add Llama4 GHA by @suiyoubi :: PR: #13442
- add memory profile interface to perf scripts by @erhoo82 :: PR: #13413
- Add fp8_param argument back to mixed precision plugin for backward compatibility by @guyueh1 :: PR: #13522
- [automodel] add find_unused_parameters=True for DDP by @akoumpa :: PR: #13366
- ci: Update success message by @ko3n1g :: PR: #13541
- [Audio] TransformerUNet: predictive model support added by @nasretdinovr :: PR: #13470
- Test Hyena mixer CP equivalency  by @farhadrgh :: PR: #13330
- use null tokenizer by @malay-nagda :: PR: #13480
- ci: Remove optional marker by @ko3n1g :: PR: #13469
- Update extra_requires and requirements by @thomasdhc :: PR: #13359
- Fix default config for LlamaNemotron Ultra by @suiyoubi :: PR: #13542
- [audio] Improve test coverage for audio losses by @anteju :: PR: #13309
- deepseek finetuning callback error change by @SDcodehub :: PR: #13483
- ci(fix): Add `__init__` to selective-triggering by @ko3n1g :: PR: #13577
- nsys profile filename ranks info by @malay-nagda :: PR: #13576
- chore: Update setup.py by @ko3n1g :: PR: #13566
- Fix Llama importer by @suiyoubi :: PR: #13583
- [automodel] fix --mbs/gbs dtype and chat-template by @akoumpa :: PR: #13602
- Reconfigure 'limit_<train|val>_batches' by @maanug-nv :: PR: #13523
- ci: Optional speech tests by @ko3n1g :: PR: #13606
- [Automodel] Fix CP device_mesh issue, use PTL distsampler by @BoxiangW :: PR: #13473
- [automodel] fix log message by @akoumpa :: PR: #13612
- Tests for evaluation with NVIDIA Evals Factory by @chtruong814 :: PR: #13627
- Fix ptl import in notebooks by @maanug-nv :: PR: #13608
- [automodel] dist.abort -> dist.destroy_process_group by @akoumpa :: PR: #13578
- Skip eval unit test by @chtruong814 :: PR: #13635
- Fix image_processor config in Energon path by @AtsunoriFujita :: PR: #13618
- Add Gemma3 VL model by @xiangxu-google :: PR: #13536
- Set L2_NeMo_2_EVAL as optional by @chtruong814 :: PR: #13644
- Update install to use pip install by @thomasdhc :: PR: #13605
- Multi node settings for evaluation nemo-run script by @athitten :: PR: #13568
- [Llama4] Fix the missing args in the recipe by @gdengk :: PR: #13649
- Bump nvidia-modelopt to 0.29.0 by @AAnoosheh :: PR: #13599
- Update README.md for 25.04 release by @snowmanwwg :: PR: #13654
- [automodel] consolidate sft peft scripts by @akoumpa :: PR: #13634
- Qwen3 by @cuichenx :: PR: #13554
- Set env variables for eval tests by @marta-sd :: PR: #13658
- build: multimodal-only by @ko3n1g :: PR: #13665
- [Audio] TransformerUNet: predictive model tests added by @nasretdinovr :: PR: #13648
- [automodel] consolidate vllm scripts by @akoumpa :: PR: #13670
- build: Pin transformers by @ko3n1g :: PR: #13675
- ci: Enable codecov checks by @ko3n1g :: PR: #13497
- ci: Add `init-file-checker` by @ko3n1g :: PR: #13684
- Add use_sharp and use user buffer registration args in perf scripts by @youngeunkwon0405 :: PR: #13521
- Remove is-optional marker for L2_NeMo_2_EVAL by @marta-sd :: PR: #13669
- gpu type and #devices CLI args by @malay-nagda :: PR: #13620
- perf scripts updates by @malay-nagda :: PR: #13456
- Use audio codec without discriminators in SpeechLM2 tests by @pzelasko :: PR: #13711
- Update changelog for `r2.3.1` by @github-actions[bot] :: PR: #13719
- Recipe default value fix for Llama4 by @suiyoubi :: PR: #13696
- build: Lift numba by @ko3n1g :: PR: #13735
- New key override for timestamps by @melllinia :: PR: #13743
- Fixed Mllama Energon config by @AtsunoriFujita :: PR: #13574
- Update convert_to_tarred_audio_dataset.py by @ssh-meister :: PR: #13755
- Enable dropout recompute in LoRA by @michal2409 :: PR: #13745
- Address VDR feedback for NeMo FW evaluations by @athitten :: PR: #13701
- remove blocks unused to increase coverage by @romanbrickie :: PR: #13511
- Fix Flux Recipe for FSDP/DDP by @suiyoubi :: PR: #13715
- Try soften protobuf version requirement by @pablo-garay :: PR: #13747
- Flux FP8 recipe by @Victor49152 :: PR: #13584
- Gemma3 Fix and Tests by @suiyoubi :: PR: #13661
- Disable local gradient checker in performance scripts by @erhoo82 :: PR: #13768
- [Audio] Tests: training for mask, pred and SB models by @nasretdinovr :: PR: #13736
- Refactor MSC integration in exp manager by @shunjiad :: PR: #13626
- [fix] vpp error in Gemma3 by @ZhiyuLi-Nvidia :: PR: #13784
- ci: Ensure approval queue fetches all CICD workflows using pagnation by @chtruong814 :: PR: #13798
- ci: make_request in approval test queue appends next url for status checks only by @chtruong814 :: PR: #13802
- Remove guard for masking tests and improve coverage by @anteju :: PR: #13787
- fix: After mcore bump by @ko3n1g :: PR: #13781
- Fix Gemma3VL training bugs by @sharanmayank :: PR: #13766
- [NeMo 2.0] Remove the restriction of load_model_state_dict for cfsdp by @shjwudp :: PR: #13512
- Add option to construct Llama model with Transformer Engine op fuser by @timmoon10 :: PR: #13776
- [Evaluation] Add support for simple-evals and tasks that require logprobs by @marta-sd :: PR: #13647
- remove stale section by @akoumpa :: PR: #13759
- fix moe_router_pre_softmax for Mixtral by @akoumpa :: PR: #13678
- fix: improve sequence length handling to fix nan in loss when turning on cudagraph by @katec846 :: PR: #13779
- Gemma3 Energon Dataset by @suiyoubi :: PR: #13813
- Rectify BLEU evaluation by @ankitapasad :: PR: #13762
- ci: Moved workflows by @ko3n1g :: PR: #13828
- ci: Moved templates by @ko3n1g :: PR: #13830
- [Build] Bump bitsandbytes dependency to 0.45.5 (ubuntu 22.04 compatibility) by @pramodk :: PR: #13789
- update for `PYTORCH_CUDA_ALLOC_CONF` env var by @malay-nagda :: PR: #13837
- [Llama4] Enable VLM Dec cudagraph  by @gdengk :: PR: #13767
- Support MSC URL in LLM checkpointing by @shunjiad :: PR: #13805
- additional metrics by @dimapihtar :: PR: #13754
- Expand modelopt version range by @chtruong814 :: PR: #13850
- Alit/nmh4b by @JRD971000 :: PR: #13481
- [Tutorial] Train your own reasoning model in 48 hours on a single GPU by @Maghoumi :: PR: #13853
- Enabled C2C-PCie bridge through NCCL by @sanandaraj5597 :: PR: #13621
- Added safe loading of models by @nithinraok :: PR: #13607
- Add NemotronH Performance Script by @guyueh1 :: PR: #13528
- Hyena SE/MR B2B Kernel integration by @farhadrgh :: PR: #13518
- chore: Destroy buildcache by @ko3n1g :: PR: #13869
- tests: Fix Qwen test by @ko3n1g :: PR: #13888
- fix: improve error handling in `is_multistorageclient_url` by @shunjiad :: PR: #13885
- feat(eval): adds benchmark adapters that allow specisal reasoning models by @agronskiy :: PR: #13709
- perf scripts 25.07 refactor by @malay-nagda :: PR: #13875
- Fix E5 and LlamaEmbedding Conversion by @suiyoubi :: PR: #13890
- Bug fix for NCCL vars by @sanandaraj5597 :: PR: #13908
- Reranker Model Support by @suiyoubi :: PR: #13876
- numa cmd in bash by @malay-nagda :: PR: #13914
- Fix BERT issue with PP by @suiyoubi :: PR: #13916
- [Llama4] Fix Vp_stage to enable VP for VLM llama4 by @gdengk :: PR: #13873
- Enable NVTX profiling in MCore by @minitu :: PR: #13820
- [Qwen3-MoE] Add Qwen3 MoE perf recipe for 30b and 235b by @gdengk :: PR: #13895
- lazy import bnbconfig by @akoumpa :: PR: #13919
- Set TRANSFORMERS_OFFLINE=1 and HF_HUB_OFFLINE=1 in CI tests by @chtruong814 :: PR: #13932
- [peft] align adapter output shape with wrapped module output shape by @guyueh1 :: PR: #13922
- [automodel] move only lora adapters to cpu by @akoumpa :: PR: #13931
- Fix vp_stage not found when fsdp by @gautham-kollu :: PR: #13817
- Fix single optional import if ModelOpt not installed by @AAnoosheh :: PR: #13923
- Revert "Set TRANSFORMERS_OFFLINE=1 and HF_HUB_OFFLINE=1 in CI tests by @chtruong814 :: PR: #13938
- Enable LoRA for TELinear layers by @cuichenx :: PR: #13929
- Freeze tags in in `r2.4.0` by @github-actions[bot] :: PR: #13945
- Cherry pick `Use jiwer less than 4.0.0 (13997)` into `r2.4.0` by @ko3n1g :: PR: #13998
- Cherry pick `Remove container license reference (14010)` into `r2.4.0` by @ko3n1g :: PR: #14017
- move classes to module to use __target__ feature by @nithinraok :: PR: #14023
- Cherry pick `bf16 grads for bf16 jobs (14016)` into `r2.4.0` by @ko3n1g :: PR: #14020
- Cherry pick `Remove nemo1 stable diffusion test (14018)` into `r2.4.0` by @ko3n1g :: PR: #14019
- Version bump to `2.4.0rc1.dev0` by @github-actions[bot] :: PR: #14047
- Cherry pick `Fix Loading Custom Quantization Config (13934)` into `r2.4.0` by @ko3n1g :: PR: #13950
- Cherry pick `[automodel] fix sft notebook (14002)` into `r2.4.0` by @ko3n1g :: PR: #14003
- Cherry pick `Use average reduction in FSDP grad reduce-scatter when grad dtype is … (13981)` into `r2.4.0` by @ko3n1g :: PR: #14004
- Cherry pick `GPU memory logging update (13982)` into `r2.4.0` by @ko3n1g :: PR: #14021
- Cherry pick `Remove kaldiio (14006)` into `r2.4.0` by @ko3n1g :: PR: #14032
- Cherry pick `Set L2_NeMo_2_Flux_Import_Test to be optional (14056)` into `r2.4.0` by @ko3n1g :: PR: #14058
- Cherry pick `Bump protobuf to 5.29.5 (14045)` into `r2.4.0` by @ko3n1g :: PR: #14060
- Cherry pick `Detect hardware before enabling DeepEP  (14022)` into `r2.4.0` by @ko3n1g :: PR: #14068
- Version bump to `2.4.0rc2.dev0` by @github-actions[bot] :: PR: #14115
- Cherry pick `Fix SFT Dataset Bug (13918)` into `r2.4.0` by @ko3n1g :: PR: #14074
- Cherry pick `Align adapter shape with base linear output shape (14009)` into `r2.4.0` by @ko3n1g :: PR: #14083
- Cherry pick `[MoE] Update the fp8 precision interface for llama4 and qwen3 (14094)` into `r2.4.0` by @ko3n1g :: PR: #14104
- Cherry pick `[Llama4] Tokenizer naming update (14114)` into `r2.4.0` by @ko3n1g :: PR: #14123
- Cherry pick `Bump to pytorch 25.05 container along with TE update (13899)` into `r2.4.0` by @ko3n1g :: PR: #14145
- Cherry pick `Perf scripts updates (14005)` into `r2.4.0` by @ko3n1g :: PR: #14129
- Cherry pick `Remove unstructured (14070)` into `r2.4.0` by @ko3n1g :: PR: #14147
- Version bump to `2.4.0rc3.dev0` by @github-actions[bot] :: PR: #14165
- Cherry pick `Add checkpoint info for NIM Embedding Expor Tutorial (14177)` into `r2.4.0` by @ko3n1g :: PR: #14178
- Cherry pick `Fix dsv3 script (14007)` into `r2.4.0` by @ko3n1g :: PR: #14182
- Cherry pick `405b perf script updates (14176)` into `r2.4.0` by @chtruong814 :: PR: #14195
- Cherry pick `Fix nemotronh flops calculator (14161)` into `r2.4.0` by @chtruong814 :: PR: #14202
- Cherry pick `Add option to disable gloo process groups` (#14156) into `r2.4.0` by @chtruong814 :: PR: #14220
- Cherry pick `Remove g2p_en (14204)` into `r2.4.0` by @chtruong814 :: PR: #14212
- Cherry pick `diffusion mock data null args (14173)` into `r2.4.0` by @chtruong814 :: PR: #14217
- Cherry pick `perf-scripts: Change b200 config to EP8 (14207)` into `r2.4.0` by @chtruong814 :: PR: #14223
- Cherry pick `Change RerankerSpecter Dataset question key (14200)` into `r2.4.0` by @chtruong814 :: PR: #14224
- Cherry pick `Fix the forward when final_loss_mask is not present (14201)` into `r2.4.0` by @chtruong814 :: PR: #14225
- Cherry pick `Fix Llama Nemotron Nano Importer (14222)` into `r2.4.0` by @chtruong814 :: PR: #14226
- Cherry pick `[automodel] fix loss_mask pad token (14150)` into `r2.4.0` by @chtruong814 :: PR: #14227
- [Performance script] FSDP-UBR related recipe update (#14208) by @youngeunkwon0405 :: PR: #14233
- Fix for MCore dist ckpt loading #14229 by @stevehuang52 :: PR: #14239
- cherry-pick fix eval beam search ctc script by @lilithgrigoryan :: PR: #14242
- Cherry pick `Moving export security fixes over here (14254)` into `r2.4.0` by @chtruong814 :: PR: #14261
- Cherry pick `Confidence fix for tutorial (14250)` into `r2.4.0` by @chtruong814 :: PR: #14266
- Cherry pick `added new models to documentation (14264)` into `r2.4.0` by @chtruong814 :: PR: #14278
- Cherry-pick `FIx Flux & Flux_Controlnet initialization issue` (#14263) into `r2.4.0` by @chtruong814 :: PR: #14273
- Cherry pick `update ffmpeg install (14237)` into `r2.4.0` by @chtruong814 :: PR: #14279

</details>

## NVIDIA Neural Modules 2.3.2

This release addresses known security issues. For the latest NVIDIA Vulnerability Disclosure Information visit <https://www.nvidia.com/en-us/security/>, for acknowledgement please reach out to the NVIDIA PSIRT team at <PSIRT@nvidia.com>

## NVIDIA Neural Modules 2.3.1

### Highlights

- Collections
  - LLM
    - Llama 4: Fixed an accuracy issue caused by MoE probability normalization. Improved pre-train and fine-tune performance.
- Export & Deploy
  - Updated vLLMExporter to use vLLM V1 to address a security vulnerability.
- AutoModel
  - Improved chat-template handling.
- Fault Tolerance
  - Local checkpointing: Fixed support for auto-inserted metric names for resuming from local checkpoints.

### Detailed Changelogs

</details>

#### Export

<details><summary>Changelog</summary>

- Cherry-pick `Update vLLMExporter to use vLLM V1` (#13498) into `r2.3.0` by @chtruong814 :: PR: #13631

</details>

#### Uncategorized

<details><summary>Changelog</summary>

- Bump to 2.3.1 by @chtruong814 :: PR: #13507
- Cherry pick `Use explicitly cached canary-1b-flash in CI tests (13237)` into `r2.3.0` by @ko3n1g :: PR: #13508
- Cherry pick `[automodel] bump liger-kernel to 0.5.8 + fallback (13260)` into `r2.3.0` by @ko3n1g :: PR: #13308
- Cherry-pick `Add recipe and ci scripts for qwen2vl` to `r2.3.0` by @romanbrickie :: PR: #13336
- Cherry pick `Fix skipme handling (13244)` into `r2.3.0` by @ko3n1g :: PR: #13376
- Cherry pick `Allow fp8 param gather when using FSDP (13267)` into `r2.3.0` by @ko3n1g :: PR: #13383
- Cherry pick `Handle boolean args for performance scripts and log received config (13291)` into `r2.3.0` by @ko3n1g :: PR: #13416
- Cherry pick `new perf configs (13110)` into `r2.3.0` by @ko3n1g :: PR: #13431
- Cherry pick `Adding additional unit tests for the deploy module (13411)` into `r2.3.0` by @ko3n1g :: PR: #13449
- Cherry pick `Adding more export tests (13410)` into `r2.3.0` by @ko3n1g :: PR: #13450
- Cherry pick `[automodel] add FirstRankPerNode (13373)` into `r2.3.0` by @ko3n1g :: PR: #13559
- Cherry pick `[automodel] deprecate global_batch_size dataset argument (13137)` into `r2.3.0` by @ko3n1g :: PR: #13560
- Cherry-pick `[automodel] fallback FP8 + LCE -> FP8 + CE`  (#13349) into `r2.3.0` by @chtruong814 :: PR: #13561
- Cherry pick `[automodel] add find_unused_parameters=True for DDP (13366)` into `r2.3.0` by @ko3n1g :: PR: #13601
- Cherry pick `Add CI test for local checkpointing (#13012)` into `r2.3.0` by @ananthsub :: PR: #13472
- Cherry pick `[automodel] fix --mbs/gbs dtype and chat-template (13598)` into `r2.3.0` by @akoumpa :: PR: #13613
- Cherry-pick `Update t5.py` (#13082) to `r2.3.0` and `bump mcore to f98b1a0`  by @chtruong814 :: PR: #13642
- [Automodel] Fix CP device_mesh issue, use PTL distsampler (#13473) by @akoumpa :: PR: #13636
- [Llama4] Fix the recipe bug - cherrypick #13649 by @gdengk :: PR: #13650
- build: Pin transformers (#13675) by @ko3n1g :: PR: #13692

</details>

## NVIDIA Neural Modules 2.3.0

### Highlights

- Export & Deploy
  - NeMo 2.0 export path for NIM
  - ONNX and TensorRT Export for NIM Embedding Container
  - In-framework deployment for HF Models
  - TRT-LLM deployment for HF Models in NeMo Framework
- Evaluation
  - Integrate nvidia-lm-eval to NeMo FW for evaluations with OpenAI API compatible in-framework deployment
- AutoModel
  - VLM AutoModelForImageForTextToText
  - FP8 for AutoModel
  - Support CP with FSDP2
  - Support TP with FSDP2
  - Performance Optimization
    - add support for cut cross entropy & liger kernel
    - Gradient Checkpointing
- Fault Tolerance
  - Integrate NVRx v0.3 Local checkpointing
- Collections
  - LLM
    - Llama4
    - Llama Nemotron Ultra
    - Llama Nemotron Super
    - Llama Nemotron Nano
    - Nemotron-h/5
    - DeepSeek V3 Pretraining
    - Evo2
    - Qwen 2.5
    - LoRA for Qwen3-32B and Qwen3-30B-A3B
  - MultiModal
    - FLUX
    - Gemma 3
    - Qwen2-VL
  - ASR
    - NeMo Run support for ASR training
    - N-Gram LM on GPU for AED
    - N-Gram LM on GPU + Transducer greedy decoding (RNN-T, TDT)
    - Timestamps support for AED timestamp supported models
    - Migrate SpeechLM to NeMo 2.0
    - Canary-1.1
    - Replace ClassificationModels class with LabelModels
- Performance
  - Functional MXFP8 support for (G)B200
  - Current scaling recipe with TP communication overlap and FP8 param gathers
  - Custom FSDP support that fully utilizes GB200 NVL72

### Detailed Changelogs

#### ASR

<details><summary>Changelog</summary>

- Added model config params for Canary-1B-Flash, Canary-180M-Flash models by @KunalDhawan :: PR: #12588
- Canary tutorial by @ankitapasad :: PR: #12613
- Canary tutorial fix timestamp by @ankitapasad :: PR: #12677
- revert config by @nithinraok :: PR: #12689
- canary longform inference script with timestamps option by @krishnacpuvvada :: PR: #12653
- Fix default timestamps value for Hybrid ASR models by @artbataev :: PR: #12681
- Fix k2 installation with PyTorch 2.6.0 by @artbataev :: PR: #12686
- Improve time and RTFx report for ASR by @artbataev :: PR: #12680
- Modify train args by @ankitapasad :: PR: #12700
- Fix asr doc warnings by @nithinraok :: PR: #12720
- Rename `FastNGramLM` -> `NGramGPULanguageModel` by @artbataev :: PR: #12755
- transcribe fix for new hypotheses by @nune-tadevosyan :: PR: #12801
- Fix timestamps when cuda graphs enabled by @monica-sekoyan :: PR: #12808
- update streaming conformer by @stevehuang52 :: PR: #12846
- AED Decoding with N-Gram LM by @artbataev :: PR: #12730
- update notebook by @nithinraok :: PR: #13088
- bugfix ASR_Context_Biasing.ipynb by @lilithgrigoryan :: PR: #13109
- Change branch for installation from main to r2.3.0 by @ankitapasad :: PR: #13266

</details>

#### TTS

<details><summary>Changelog</summary>

- Add Magpie-TTS and Updates NeMo Audio Codecs by @blisc :: PR: #12606
- fix bug from prior commit (#13264) by @blisc :: PR: #13328

</details>

#### NLP / NMT

<details><summary>Changelog</summary>

- Remove old peft docs by @cuichenx :: PR: #12675
- Add code coverage for llm gpt models conversion tests by @suiyoubi :: PR: #12665
- Make BERT TransformerBlockWithPostLNSupport accept more inputs from Mcore by @suiyoubi :: PR: #12685
- remove gifs from documentation by @dimapihtar :: PR: #12732
- Rename `FastNGramLM` -> `NGramGPULanguageModel` by @artbataev :: PR: #12755
- fix NeMo documentation by @dimapihtar :: PR: #12754
- GPT Model/Data/Recipe Unit Test by @suiyoubi :: PR: #12757
- ci: Exclude nlp, mm, vision collections by @ko3n1g :: PR: #12816
- Add vocab size as attr to GPT and T5 Configs, use file name based logger in llm.gpt.data by @hemildesai :: PR: #12862
- Fix transformer layer api with megatron cbc89b3 by @yaoyu-33 :: PR: #12885

</details>

#### Text Normalization / Inverse Text Normalization

<details><summary>Changelog</summary>

- Rename `FastNGramLM` -> `NGramGPULanguageModel` by @artbataev :: PR: #12755

</details>

#### Export

<details><summary>Changelog</summary>

- GHA Conversion Test and Importer/Exporter Refactor by @suiyoubi :: PR: #12597
- Fix Llama Embedding Model Exporting keys by @suiyoubi :: PR: #12691
- build: Add trtllm by @ko3n1g :: PR: #12672
- Fix trt-llm install by @chtruong814 :: PR: #12827
- Update LLaVA's next HF exporter to load ViT checkpoint from YAML by @eagle705 :: PR: #12841
- Support huggingface export to tensorrtllm by @pthombre :: PR: #12889
- Adds a built stage for the trt-llm wheel to reduce the overall test image size by @chtruong814 :: PR: #12883

</details>

#### Uncategorized

<details><summary>Changelog</summary>

- Update changelog-build.yml by @ko3n1g :: PR: #12584
- Update changelog for `r2.2.0` by @github-actions[bot] :: PR: #12585
- Add comments for requirements by @thomasdhc :: PR: #12603
- [automodel] FSDP2Strategy: move to device if using a single-device by @akoumpa :: PR: #12593
- build: Remove numba pin by @ko3n1g :: PR: #12604
- docs: Update installation guides by @ko3n1g :: PR: #12596
- Change Llama Scaling Factor type to Float by @suiyoubi :: PR: #12616
- ci: Test multiple python versions by @ko3n1g :: PR: #12619
- ci: Disable reformat by @ko3n1g :: PR: #12620
- Updating ModelOpt to 0.25.0 by @janekl :: PR: #12633
- [automodel] add additional hf_dataset tests by @akoumpa :: PR: #12646
- [automodel] add jit_transform tests by @akoumpa :: PR: #12645
- [automodel] init eos_token_id inside data module by @yuanzhedong :: PR: #12610
- [automodel] grad ckpt by @akoumpa :: PR: #12644
- bugfix(llm/LLaMa) - dropout_position can never be equal to extended string by @soluwalana :: PR: #12649
- Fix inference pipeline quality issue by @Victor49152 :: PR: #12639
- [automodel] switch to direct=True to propage return codes in nemorun by @akoumpa :: PR: #12651
- add Auto Conf support for bert, t5, qwen, starcoder models by @dimapihtar :: PR: #12601
- ci: Upload coverage by @ko3n1g :: PR: #12668
- ci: Re-enable changed-files action by @ko3n1g :: PR: #12683
- build: Pin sox by @ko3n1g :: PR: #12701
- add neva quantization by @linnanwang :: PR: #12698
- Clip coverage by @abhinavg4 :: PR: #12696
- GHA CI test: Remove unnecessary directive by @pablo-garay :: PR: #12714
- minor perf fixes by @malay-nagda :: PR: #12656
- Add DeepSeek V2 Lite into llm __init__.py by @suiyoubi :: PR: #12664
- Add Llama-Nemotron Nano and 70B models by @suiyoubi :: PR: #12712
- Save batch norm running stats in PEFT checkpoints by @cuichenx :: PR: #12666
- Fix document Readme under nemo to add more information by @yaoyu-33 :: PR: #12699
- Fix ub_overlap_ag by @cuichenx :: PR: #12721
- Toggle fast tokenizer if error occurs by @cuichenx :: PR: #12722
- Update README.md for blackwell and AutoModel by @snowmanwwg :: PR: #12612
- Raise error on import_ckpt with overwrite=False plus README for checkpoint_converters by @janekl :: PR: #12693
- [automodel] fix validation_step by @soluwalana :: PR: #12659
- [automodel] vlm tests by @akoumpa :: PR: #12716
- Auto Configurator code coverage by @dimapihtar :: PR: #12694
- [automodel] fix automodle benchmark script by @yuanzhedong :: PR: #12605
- Remove unnecessary directives by @pablo-garay :: PR: #12743
- Add recipe tests for coverage by @cuichenx :: PR: #12737
- Add Qwen2.5 in NeMo2 by @suiyoubi :: PR: #12731
- add fallback_module to safe_import_from by @akoumpa :: PR: #12726
- Update quantization scripts & relax modelopt requirement specifier by @janekl :: PR: #12709
- Import guard fasttext by @thomasdhc :: PR: #12758
- [automodel] chunked cross entropy by @akoumpa :: PR: #12752
- Add fsdp automodel test by @BoxiangW :: PR: #12718
- [automodel] if peft move only adapters to cpu by @akoumpa :: PR: #12735
- [automodel] update hf mockdataset by @akoumpa :: PR: #12643
- [automodel] remove unused cell in multinode notebook by @yuanzhedong :: PR: #12624
- Yash/llava next coverage by @yashaswikarnati :: PR: #12745
- Tidy code: remove unneeded statements/lines by @pablo-garay :: PR: #12771
- Pass tensor instead of raw number in _mock_loss_function in PTQ by @janekl :: PR: #12769
- ci: Run on nightly schedule by @ko3n1g :: PR: #12775
- Add logs for checkpoint saving start and finalization by @lepan-google :: PR: #12697
- Alit/test coverage by @JRD971000 :: PR: #12762
- Fix loss mask with packed sequence by @ashors1 :: PR: #12642
- Add pruning recipe by @kevalmorabia97 :: PR: #12602
- Update qwen2-v1 to use NeMo quick_gelu by @thomasdhc :: PR: #12787
- [doc] Fixes for audio doc warnings by @anteju :: PR: #12736
- ci: Measure multiprocessing by @ko3n1g :: PR: #12778
- ci: Fix flaky LLM tests by @ko3n1g :: PR: #12807
- Add BERT/Qwen2.5 Unit test and Refactor all GHA Conversion Tests by @suiyoubi :: PR: #12785
- Fix TransformerBlock cuda_graphs compatibility with MCore by @buptzyb :: PR: #12779
- ci: Remove `--branch` by @ko3n1g :: PR: #12809
- ci: Move scripts fully down to files by @ko3n1g :: PR: #12802
- add __init__.py to make this a package by @akoumpa :: PR: #12814
- Update changelog for `r2.2.1` by @github-actions[bot] :: PR: #12818
- add finetune support for Auto Configurator by @dimapihtar :: PR: #12770
- [automodel] add cpu:gloo to backend by @akoumpa :: PR: #12832
- add missing call to _apply_liger_kernel_to_instance by @akoumpa :: PR: #12806
- Prune docker images in GHA older than 8hrs by @chtruong814 :: PR: #12838
- [audio] Adding tests for predictive models by @anteju :: PR: #12823
- Update resiliency example notebook readme and add links to the brev launchable by @ShriyaRishab :: PR: #12843
- [automodel] qlora peft  by @yzhang123 :: PR: #12817
- ci: Increase prune time by @ko3n1g :: PR: #12860
- Update base container in `Dockerfile.speech` by @artbataev :: PR: #12859
- Fix qwen2.5 1.5b configuration inheritance bug by @Aprilistic :: PR: #12852
- Update modelopt upperbound to 0.27 by @thomasdhc :: PR: #12788
- Non-blocking checkpoint cleanup failure by @jstjohn :: PR: #12804
- Improve evo2 dataset test and testability by @jstjohn :: PR: #12857
- Expand test converage neva / mllama by @yaoyu-33 :: PR: #12715
- Weekly bump by @ko3n1g :: PR: #12891
- ci: Optional_L2_NeMo_2_SSM_Finetuning by @ko3n1g :: PR: #12893
- docs: Update guide to PEP508 by @ko3n1g :: PR: #12890
- Replace lm-eval with nvidia-lm-eval by @chtruong814 :: PR: #12888
- Handle CUDA_DEVICE_MAX_CONNECTIONS before job launch by @guyueh1 :: PR: #12833
- add nemotron5 by @JRD971000 :: PR: #12660
- Bump vllm 0.8.2 by @Laplasjan107 :: PR: #12753
- DeepseekV3 SFT finetuning perf config  by @gdengk :: PR: #12829
- add apply_chat_template method to TokenizerSpec + AutoTokenizer by @akoumpa :: PR: #12878
- add accelerate to dependencies by @akoumpa :: PR: #12871
- [automodel] Add FSDPv2-compatible context parallelism support. by @cspades :: PR: #12821
- [fault tolerance] Add local checkpointing support by @ananthsub :: PR: #12839
- ci: Bump release-freeze by @ko3n1g :: PR: #12914
- ci: Use PAT for code-freeze by @ko3n1g :: PR: #12915
- ci: Use correct environment by @ko3n1g :: PR: #12917
- Freeze tags in in `r2.3.0` by @github-actions[bot] :: PR: #12919
- chore: Bump version to 2.3.0.rc2 by @chtruong814 :: PR: #12920
- Version bump to `2.3.0rc3.dev0` by @github-actions[bot] :: PR: #12921
- Cherry pick `[automodel] Add linear ce loss support (12825)` into `r2.3.0` by @ko3n1g :: PR: #12922
- Cherry pick `DeepSeek V3 Multi Token Prediction (12550)` into `r2.3.0` by @ko3n1g :: PR: #12928
- Cherry pick `Set L2_NeMo_2_EVAL test to be optional (12949)` into `r2.3.0` by @ko3n1g :: PR: #12951
- Cherry pick `GB200 LLM performance scripts tuning (12791)` into `r2.3.0` by @ko3n1g :: PR: #12923
- Cherry pick `Allow configuration of PP communication backend to UCC in nemo2 (11755)` into `r2.3.0` by @ko3n1g :: PR: #12946
- Cherry pick `guard bitsandbytes based on cuda availability (12937)` into `r2.3.0` by @ko3n1g :: PR: #12958
- Cherry pick `Hugging Face model deployment support (12628)` into `r2.3.0` by @ko3n1g :: PR: #12962
- Cherry pick `fix macro-acc for pair-audio eval (12908)` into `r2.3.0` by @ko3n1g :: PR: #12963
- Cherry pick `Add energon dataset support for Qwen2VL (12831)` into `r2.3.0` by @ko3n1g :: PR: #12966
- Cherry pick `Make TETransformerLayerAutocast Support Cuda Graph (12075)` into `r2.3.0` by @ko3n1g :: PR: #12967
- Cherry pick `Use nvidia-lm-eval for evaluation (12902)` into `r2.3.0` by @ko3n1g :: PR: #12971
- Cherry pick `[NeMo 2.0] Interface for using MXFP8 and FP8 current scaling recipes (12503)` into `r2.3.0` by @ko3n1g :: PR: #12974
- Cherry pick `Fix trtllm and lightning conflict (12943)` into `r2.3.0` by @ko3n1g :: PR: #12981
- Cherry pick `Update v3 finetuning recipe (12950)` and `Specify PP first/last in strategy (12992)` into `r2.3.0` by @ko3n1g :: PR: #12984
- Cherry pick `Resolve an issue in custom megatron FSDP config setting (12948)` into `r2.3.0` by @ko3n1g :: PR: #12987
- Cherry pick `Remove getattr_proxy to avoid problematic edge cases (12176)` into `r2.3.0` by @ko3n1g :: PR: #12990
- Cherry pick `Enable async requests for in-fw deployment with OAI compatible server (12980)` into `r2.3.0` by @ko3n1g :: PR: #12994
- Cherry pick `initialize model with metadata (12496)` into `r2.3.0` by @ko3n1g :: PR: #12997
- Cherry pick `Bugfix for logits support for hf deployment (12965)` into `r2.3.0` by @ko3n1g :: PR: #13001
- Cherry pick `Update nvidia-resiliency-ext to be >= 0.3.0 (12925)` into `r2.3.0` by @ko3n1g :: PR: #13000
- Cherry-pick Fix params_dtype for distillation and GPT HF Exporter head_dim for pruning to r2.3.0 by @kevalmorabia97 :: PR: #13002
- Install nvidia-pytriton on arm (#13011) by @thomasdhc :: PR: #13013
- Version bump to `2.3.0rc4.dev0` by @github-actions[bot] :: PR: #13041
- Cherry pick `Alit/nemotron h (12942)` into `r2.3.0` by @ko3n1g :: PR: #13007
- Cherry pick `[Automodel] Add TP/SP support with default llama-like sharding plan (12796)` into `r2.3.0` by @ko3n1g :: PR: #13017
- Cherry pick `Add initial docs broken link check (12977)` into `r2.3.0` by @ko3n1g :: PR: #13045
- Cherry pick `Fix MoE Init to not use Bias in test_strategy_lib.py (13009)` into `r2.3.0` by @ko3n1g :: PR: #13014
- Cherry pick `cleaner tflops log name (13005)` into `r2.3.0` by @ko3n1g :: PR: #13024
- Cherry pick `Improve t5 test coverage (12803)` into `r2.3.0` by @ko3n1g :: PR: #13025
- Cherry pick `put the warning on the right place (12909)` into `r2.3.0` by @ko3n1g :: PR: #13035
- Cherry pick `Temporary disable CUDA graphs in DDP mode for transducer decoding (12907)` into `r2.3.0` by @ko3n1g :: PR: #13036
- Cherry pick `[automodel] peft fix vlm (13010)` into `r2.3.0` by @ko3n1g :: PR: #13037
- Cherry pick `Only run the docs link check on the container (13068)` into `r2.3.0` by @ko3n1g :: PR: #13070
- Cherry pick `Add fp8 recipe option to perf script (13032)` into `r2.3.0` by @ko3n1g :: PR: #13055
- Cherry pick `Unified ptq export (12786)` into `r2.3.0` by @ko3n1g :: PR: #13062
- Cherry pick `Fix VP list index out of range from Custom FSDP (13021)` into `r2.3.0` by @ko3n1g :: PR: #13077
- Cherry pick `Add logging to cancel out PTL's warning about dataloader not being resumable (13072)` into `r2.3.0` by @ko3n1g :: PR: #13100
- Cherry pick `Fix long sequence generation after new arg introduced in mcore engine (13049)` into `r2.3.0` by @ko3n1g :: PR: #13104
- Cherry pick `Support Mamba models quantization (12631)` into `r2.3.0` by @ko3n1g :: PR: #13105
- Cherry pick `Add track_io to user buffer configs (13071)` into `r2.3.0` by @ko3n1g :: PR: #13111
- ci: Onboard 8-GPU runner (#13115) by @ko3n1g :: PR: #13121
- Cherry pick `Add fine-tuning dataset function for FineWeb-Edu and update automodel… (13027)` into `r2.3.0` by @ko3n1g :: PR: #13118
- Cherry pick `Re-add sox to asr requirements (13092)` into `r2.3.0` by @ko3n1g :: PR: #13120
- Cherry pick `Update Mllama cross attn signature to match update MCore (13048)` into `r2.3.0` by @ko3n1g :: PR: #13122
- Cherry pick `Fix Exporter for baichuan and chatglm (13095)` into `r2.3.0` by @ko3n1g :: PR: #13126
- ci: Faster builds (#13142) by @ko3n1g :: PR: #13144
- Version bump to `2.3.0rc5.dev0` by @github-actions[bot] :: PR: #13146
- ci: Fix mcore install in test container (#13152) by @ko3n1g :: PR: #13159
- ci: Fix race-condition of container setup (#13162) by @ko3n1g :: PR: #13163
- Cherry pick `Guard decord and triton import (12861)` into `r2.3.0` by @ko3n1g :: PR: #13132
- Cherry pick `Bump TE version and apply patch (13087)` into `r2.3.0` by @ko3n1g :: PR: #13139
- Cherry pick `Update Llama-Minitron pruning-distillation notebooks from NeMo1 to NeMo2 + NeMoRun (12968)` into `r2.3.0` by @ko3n1g :: PR: #13141
- Cherry pick `Export and Deploy Tests  (13076)` into `r2.3.0` by @ko3n1g :: PR: #13150
- Cherry pick `ub fp8 h100 fixes (13131)` into `r2.3.0` by @ko3n1g :: PR: #13153
- Cherry pick `Fix Transducer Decoding with CUDA Graphs in DDP with Mixed Precision (12938)` into `r2.3.0` by @ko3n1g :: PR: #13154
- Cherry pick `build: Pin modelopt (13029)` into `r2.3.0` by @chtruong814 :: PR: #13170
- Cherry pick `add fixes for nemotron-h` (13073) into `r2.3.0` by @JRD971000 :: PR: #13165
- Add dsv3 pretrain script, support flops calculation (previous #12947) by @guyueh1 :: PR: #13186
- ci: Allow running CI on weekly bump branch by @ko3n1g :: PR: #13233
- Cherry pick `Add Llama Nemotron Super/Ultra models (13044)` into `r2.3.0` by @ko3n1g :: PR: #13212
- Cherry pick `Add Blockwise FP8 to PTQ & EP to modelopt resume (12670)` into `r2.3.0` by @ko3n1g :: PR: #13239
- Cherry pick `[OAI Serving] Validate greedy generation args (redo) (13216)` into `r2.3.0` by @ko3n1g :: PR: #13242
- Cherry pick `drop sample_alpha in speechlm (13208)` into `r2.3.0` by @ko3n1g :: PR: #13246
- Cherry pick `[Eval bugfix] Move global eval-related imports inside the evaluate function (13166)` into `r2.3.0` by @ko3n1g :: PR: #13249
- Cherry pick `[Eval bugfix] Change default val of parallel_requests in eval script (13247)` into `r2.3.0` by @ko3n1g :: PR: #13253
- Cherry pick `Add tutorial for evaluation with Evals Factory (13259)` into `r2.3.0` by @ko3n1g :: PR: #13271
- Cherry pick `Fix default token durations (13168)` into `r2.3.0` by @ko3n1g :: PR: #13261
- Cherry pick `[Evaluation] Add support for nvidia-lm-eval==25.04 (13230)` into `r2.3.0` by @ko3n1g :: PR: #13274
- Cherry pick `[bug fix] set inference max seq len in inference context (13245)` into `r2.3.0` by @ko3n1g :: PR: #13276
- Cherry pick `More export and deploy unit tests (13178)` into `r2.3.0` by @ko3n1g :: PR: #13283
- Cherry pick `Reopen 13040 (13199)` into `r2.3.0` by @ko3n1g :: PR: #13303
- Cherry pick `Fix nemo1's neva notebook (13218)` into `r2.3.0` by @ko3n1g :: PR: #13312
- Cherry pick `build: various bumps (13285)` into `r2.3.0` by @ko3n1g :: PR: #13313
- Cherry-pick `ci: Increase cache pool` into `r2.3.0` by @chtruong814 :: PR: #13317
- Cherry pick `update num nodes in deepseek v3 finetune recipe (13314)` into `r2.3.0` by @ko3n1g :: PR: #13316
- Cherry pick `Fix neva notebook (13334)` into `r2.3.0` by @ko3n1g :: PR: #13335
- Cherry-pick `Add Llama4 Scout and Maverick Support (#12898)` by @ko3n1g :: PR: #13331
- Cherry pick `Fix handling Llama Embedding dimensions param and prompt type in the ONNX export tutorial (13262)` into `r2.3.0` by @ko3n1g :: PR: #13326
- Cherry-pick `Fix transformer offline for CI/CD llama4 tests` (#13339) to `r2.3.0` by @chtruong814 :: PR: #13340
- Fix llama4 test names by @chtruong814 :: PR: #13358
- Cherry pick `vLLM==0.8.5 update  (13350)` into `r2.3.0` by @ko3n1g :: PR: #13354
- Cherry-pick a test and doc fix to r2.3.0 by @chtruong814 :: PR: #13338
- Cherry pick `Add llama4 training recipe (12952)` into `r2.3.0` by @ko3n1g :: PR: #13386

</details>

## NVIDIA Neural Modules 2.2.1

### Highlights

- Training
  - Fix MoE based models training instability.
  - Fix bug in Llama exporter for Llama 3.2 1B and 3B.
  - Fix bug in LoRA linear_fc1adapter when different TP is used during saving and loading the adapter checkpoint.

### Detailed Changelogs

</details>

#### Uncategorized

<details><summary>Changelog</summary>

- Re-add reverted commits after 2.2.0 and set next version to be 2.2.1 by @chtruong814 :: PR: #12587
- Cherry pick `Fix exporter for llama models with shared embed and output layers (12545)` into `r2.2.0` by @ko3n1g :: PR: #12608
- Cherry pick `Fix TP for LoRA adapter on`linear_fc1`(12519)` into `r2.2.0` by @ko3n1g :: PR: #12607
- Bump mcore to use 0.11.1 by @chtruong814 :: PR: #12634

</details>

## NVIDIA Neural Modules 2.2.0

### Highlights

- Training
  - Blackwell and Grace Blackwell support
  - Pipeline parallel support for distillation
  - Improved NeMo Framework installation
- Export & Deploy
  - vLLM export for NeMo 2.0
- Evaluations
  - Integrate lm-eval-harness
- Collections
  - LLM
    - DAPT Example and best practices in nemo 2.0
    - [NeMo 2.0] Enable Tool Learning and add a tutorial
    - Support GPT Embedding Model (Llama 3.2 1B/3B)
    - Qwen2.5, Phi4 (via AutoModel)
    - SFT for Llama 3.3 model (via AutoModel)
    - Support BERT Embedding Model with NeMo 2.0
    - DeepSeek SFT & PEFT Support
  - MultiModal
    - Clip
    - SP for NeVA
    - CP for NeVA
    - Intern-VIT
- Automodel
  - Preview release.
  - PEFT and SFT support for LLMs available via Hugging Face’s AutoModelForCausalLM.
  - Support for Hugging Face-native checkpoints (full model and adapter only).
  - Support for distributed training via DDP and FSDP2.
- ASR/TTS
  - Lhotse: TPS-free 2D bucket estimation and filtering
  - Update model outputs to make all asr outputs to be in consistent format
  - Sortformer Release Model

### Detailed Changelogs

#### ASR

<details><summary>Changelog</summary>

- removed the line which caused a problem in nfa_tutorial by @Ssofja :: PR: #11710
- TPS-free 2D bucket estimation and filtering by @pzelasko :: PR: #11738
- Update transcribe_utils.py by @stevehuang52 :: PR: #11984
- Sortformer Diarizer 4spk v1 model PR Part 4: Sortformer Documents and Notebook Tutorials by @tango4j :: PR: #11707
- fix the issue during batched inference of Sortformer diarizer by @tango4j :: PR: #12047
- changed asr models outputs to be consistent by @Ssofja :: PR: #11818
- chore: Update notebooks by @ko3n1g :: PR: #12161
- add ctc segmentation by @ko3n1g :: PR: #12312
- clean up VAD tutorial by @stevehuang52 :: PR: #12410
- copy from main by @nithinraok :: PR: #12423
- ci: Disable ASR tests for now (#12443) by @ko3n1g :: PR: #12466
- ASR_CTC_Language_Finetuning.ipynb bugfix by @lilithgrigoryan :: PR: #12538

</details>

#### TTS

<details><summary>Changelog</summary>

- Add New Transformer Backbone for TTS Models by @blisc :: PR: #11911
- changed asr models outputs to be consistent by @Ssofja :: PR: #11818
- chore: Update notebooks by @ko3n1g :: PR: #12161

</details>

#### NLP / NMT

<details><summary>Changelog</summary>

- Use explicit imports from megatronllm_deployable.py by @janekl :: PR: #11705
- Bug fix minor bug in TRT-LLM deployment by @oyilmaz-nvidia :: PR: #11714
- gpt moe perf scripts by @malay-nagda :: PR: #11760
- Bump mcore by @ko3n1g :: PR: #11740
- Enable packed seqs for validation by @jiemingz :: PR: #11748
- Revert Mcore update since it caused regression by @pablo-garay :: PR: #11791
- Fix Gemma2 Attention Init Args by @suiyoubi :: PR: #11792
- Add null tokenizer by @erhoo82 :: PR: #11789
- Fix DistCP inference issue by @suiyoubi :: PR: #11801
- Add BERT Embedding Models E5 Recipe by @suiyoubi :: PR: #11787
- Add rope scaling configs for NeMo 1 by @BoxiangW :: PR: #11807
- Fix calculating num_available_samples by @huvunvidia :: PR: #11830
- fix sentencepiece tokenizer special tokens by @akoumpa :: PR: #11811
- add chat sft dataset to support agent tool calling by @chenrui17 :: PR: #11759
- Revert "Revert Mcore update since it caused regression (#11791)" by @ko3n1g :: PR: #11799
- fix checkpoint load issue by @dimapihtar :: PR: #11859
- Fix nemo 1 packed sequence TE version error by @cuichenx :: PR: #11874
- enable loading older TE checkpoints by @dimapihtar :: PR: #11930
- ci: Use single runner machines for unit tests by @ko3n1g :: PR: #11937
- llm performance scripts  by @malay-nagda :: PR: #11736
- [MoE] add expert tensor parallelism support for NeMo2.0 MoE by @gdengk :: PR: #11880
- add exception when loading ckpt saved by TE < 1.13 by @dimapihtar :: PR: #11988
- remove renormalize_blend_weights flag by @dimapihtar :: PR: #11975
- Llama3.2 1B Embedding Model Support by @suiyoubi :: PR: #11909
- Weekly bump by @ko3n1g :: PR: #11896
- Debug Apex distributed optimizer to handle Transformer Engine 2.0 by @timmoon10 :: PR: #12004
- throw MegatronOptimizerModule warning only with mcore models by @akoumpa :: PR: #12085
- fix nmt dataclass issue by @dimapihtar :: PR: #12081
- Propogate dp last changes from mcore by @ryantwolf :: PR: #12012
- Add error message when downloading failed. by @yuanzhedong :: PR: #12139
- interface for asymmetric pipeline schedule by @erhoo82 :: PR: #12039
- chore: Update notebooks by @ko3n1g :: PR: #12161
- Cherrypick #12382, #12415 and #12424 by @cuichenx :: PR: #12425
- ASR_CTC_Language_Finetuning.ipynb bugfix by @lilithgrigoryan :: PR: #12538

</details>

#### Text Normalization / Inverse Text Normalization

<details><summary>Changelog</summary>

- surface attn_implementation option by @akoumpa :: PR: #11873
- attn_implementation eager fallback by @akoumpa :: PR: #12060

</details>

#### NeMo Tools

<details><summary>Changelog</summary>

- build: Add `sox` to SDE by @ko3n1g :: PR: #11882
- add ctc segmentation by @ko3n1g :: PR: #12312

</details>

#### Export

<details><summary>Changelog</summary>

- Bug fix minor bug in TRT-LLM deployment by @oyilmaz-nvidia :: PR: #11714
- In-framework deployment NeMo 2.0 nemo_export.py test by @janekl :: PR: #11749
- Fix starcoder2 missing bias in nemo2 config for TRTLLM by @meatybobby :: PR: #11809
- Autodetect dtype on exporting to TensorRT-LLM by @janekl :: PR: #11907
- PTQ & TRT-LLM updates related to upcoming PyTorch 25.01 bump by @janekl :: PR: #11941
- Run Flake8 for nemo.export module by @janekl :: PR: #11728
- Skip initialization in hf export by @cuichenx :: PR: #12136
- update export io call by @akoumpa :: PR: #12144
- add default kwargs for trtllm model runner by @pablo-garay :: PR: #12248
- cherry-pick: fix[export]: reshard model correctly handles extra_state when it's a tensor (#12132) by @terrykong :: PR: #12335

</details>

#### Bugfixes

<details><summary>Changelog</summary>

- added required instalation for sox to process mp3 file by @Ssofja :: PR: #11709
- removed the line which caused a problem in nfa_tutorial by @Ssofja :: PR: #11710
- Bug fix minor bug in TRT-LLM deployment by @oyilmaz-nvidia :: PR: #11714

</details>

#### Uncategorized

<details><summary>Changelog</summary>

- Allow using vocab size from config by @shanmugamr1992 :: PR: #11718
- Fix baseline recipes by @erhoo82 :: PR: #11725
- Update changelog for `r2.1.0` by @github-actions[bot] :: PR: #11745
- ci: Fix changelog generator by @ko3n1g :: PR: #11744
- Fix 'http_port' parameter name in DeployPyTriton usages and update .qnemo compress=True path by @janekl :: PR: #11747
- Conversion NeMo and HF checkpoint script for T5 by @huvunvidia :: PR: #11739
- Add BERT Embedding Models by @suiyoubi :: PR: #11737
- Add server ready check before starting evaluation by @athitten :: PR: #11731
- only install bitsandbytes on x86 by @akoumpa :: PR: #11781
- [Bugfix] Skip processing if extra_state loads as None by @janekl :: PR: #11778
- chore(beep boop 🤖): Bump `MCORE_TAG=4dc8977...` (2025-01-07) by @ko3n1g :: PR: #11768
- make progress printer compatible with PTL v2.5.0 by @ashors1 :: PR: #11779
- Fix Mistral Conversion Issue by @suiyoubi :: PR: #11786
- build: Fix build-arg by @ko3n1g :: PR: #11815
- Lora ckpt in HF format for NeMo AutoModel by @oyilmaz-nvidia :: PR: #11712
- 8x22b seq len by @malay-nagda :: PR: #11788
- Bugfix for output_generation_logits in tensorrtllm by @athitten :: PR: #11820
- handle mistralai/Mistral-7B-Instruct-v0.3 tokenizer correctly by @akoumpa :: PR: #11839
- remove tensorstore pin in requirements*.txt by @pstjohn :: PR: #11777
- Do not load context for model transform in llm inference by @hemildesai :: PR: #11751
- update nemo2sftpeft tutorial container verison by @HuiyingLi :: PR: #11832
- Latest News updated for Cosmos by @lbliii :: PR: #11806
- Removes tensorstore 0.1.45 pin from requirements_deploy.txt by @pstjohn :: PR: #11858
- ci: Prune dangling images by @ko3n1g :: PR: #11885
- Disable tests that download datasets from web by @akoumpa :: PR: #11878
- Add context_logits for eval accuracy calculation in case of multi token prediction tasks by @athitten :: PR: #11753
- add dataset_root to SpecterDataModule by @suiyoubi :: PR: #11837
- Support both Path and str for APIs by @maanug-nv :: PR: #11865
- Run nsys callback on GBS not on MBS by @akoumpa :: PR: #11861
- ci: Set bump-branch to weekly by @ko3n1g :: PR: #11889
- chore: Update mcore-tag-bump-bot.yml by @ko3n1g :: PR: #11891
- ci: Bump Mcore in weekly PR by @ko3n1g :: PR: #11897
- check restore_config first by @akoumpa :: PR: #11890
- LinearAdapter: propagate args to _init_adapter by @akoumpa :: PR: #11902
- NeMo 2.0 fp8 conversion by @Laplasjan107 :: PR: #11845
- nemo ux expert tensor parallel by @akoumpa :: PR: #11903
- Add CP support to Neva in NeMo2 by @yaoyu-33 :: PR: #11850
- build: Move dependencies by @ko3n1g :: PR: #11790
- Add Flux and Flux Controlnet Support to Diffusion folder by @Victor49152 :: PR: #11794
- ci: Adjust bump mcore workflow by @ko3n1g :: PR: #11918
- ci: Small fix to bump workflow by @ko3n1g :: PR: #11919
- Revert #11890 and add a test that would have caught the error by @cuichenx :: PR: #11914
- ci: Adjust input argument by @ko3n1g :: PR: #11921
- Create test_phi3.py by @mayani-nv :: PR: #11843
- Enable NeMo importer and loading dist CKPT for training  by @Victor49152 :: PR: #11927
- build: Pin `triton` by @ko3n1g :: PR: #11938
- Add sharding for speechlm and vlm by @BoxiangW :: PR: #11876
- Update torch load for load from disk by @thomasdhc :: PR: #11963
- Add options to add mp_policy and parallel_fn for NeMo automodel fsdp2 by @BoxiangW :: PR: #11956
- ci: Add coverage reports by @ko3n1g :: PR: #11912
- Add batching support for evaluation by @athitten :: PR: #11934
- add use_fast option by @akoumpa :: PR: #11976
- improve error and debug messages in model connector by @cuichenx :: PR: #11979
- [checkpoint][docs] Fix typos in dist checkpointing docs by @ananthsub :: PR: #11983
- callbacks and bf16 grad by @malay-nagda :: PR: #11985
- remove --disable-ckpt from tests by @akoumpa :: PR: #11996
- nemo automodel sft squad data prep fix by @akoumpa :: PR: #11994
- Introduce evaluation API by @Glorf :: PR: #11895
- Remove deprecated tests/infer_data_path.py by @janekl :: PR: #11997
- Checkpoint saving for automodels via ModelCheckpoint by @akoumpa :: PR: #11998
- Mask vocab padding token ids from CE loss by @maanug-nv :: PR: #11999
- Add the NeMo2 memory profiling plugin by @gdengk :: PR: #12009
- chore(ci): Disable VMs cron job on forks by @mikemckiernan :: PR: #12020
- Adding speechlm AutoModel test by @oyilmaz-nvidia :: PR: #11990
- minor fix and simplify by @akoumpa :: PR: #12007
- ci: Build wheel workflow by @ko3n1g :: PR: #12021
- ci: Release workflow by @ko3n1g :: PR: #12022
- Version bump to `2.2.0rc1` by @github-actions[bot] :: PR: #12023
- ci: Run unit tests on main by @ko3n1g :: PR: #11986
- [Audio] Fix extra step in Euler sampler for flow matching inference by @racoiaws :: PR: #11989
- Set zarr range to >=2.18.2 and <3.0.0 by @chtruong814 :: PR: #12005
- ci: Run linting per domain by @ko3n1g :: PR: #12027
- Replace reference of requirements_infer.txt with requirements_deploy.txt by @chtruong814 :: PR: #12029
- ci: Always run linting by @ko3n1g :: PR: #12035
- ci: Retry on timeout by @ko3n1g :: PR: #11974
- [MoE] fix run err in mixtral22B recipe and update its perf config by @gdengk :: PR: #12036
- Version bump to `2.2.0rc2.dev0` by @github-actions[bot] :: PR: #12040
- ci: Update weekly brain by @ko3n1g :: PR: #12043
- ci: Update workflow by @ko3n1g :: PR: #12044
- nemo-automodel: fsdp2 support for peft by @akoumpa :: PR: #12008
- fix llama-3.1 hf model_id by @AtsunoriFujita :: PR: #11774
- Clip Model in Nemo2 by @abhinavg4 :: PR: #11980
- Adding TFLOPs callback for Multimodal models and NeVA calculator by @parthmannan :: PR: #11969
- ci: Allow skipping docs by @ko3n1g :: PR: #12048
- avoid missmatch error when loading older TE checkpoints by @dimapihtar :: PR: #12028
- Add padding in mllama vision encoder to align with HF by @meatybobby :: PR: #11808
- chore: Add warning for rebase by @ko3n1g :: PR: #12061
- ci: Lint Python files only by @ko3n1g :: PR: #12064
- Recipe changes for performance by @guyueh1 :: PR: #11763
- Pipeline-parallel support for Knowledge Distillation (NeMo 2) by @AAnoosheh :: PR: #11766
- add cp_comm_type param to Mistral config by @dimapihtar :: PR: #12049
- Conformer-based spectrogram estimator by @anteju :: PR: #12002
- Adding nemo CI by @abhinavg4 :: PR: #12052
- Update optimization features readme from nemo1 to nemo2 by @yaoyu-33 :: PR: #12071
- Add Llama Embedding Tutorial by @suiyoubi :: PR: #12042
- Fix Linting by @suiyoubi :: PR: #12079
- Fix hf_dataset bug by @BoxiangW :: PR: #12072
- set TOKENIZERS_PARALLELISM=True by @akoumpa :: PR: #12083
- minor fix in model's summary identation during logging by @akoumpa :: PR: #12084
- Refactor VLM modules / Add InternVit submodule support by @yaoyu-33 :: PR: #11851
- Fix SBERT with sequence_len_offset by @suiyoubi :: PR: #12057
- ci: codecov by @ko3n1g :: PR: #12030
- build: Improve installer by @ko3n1g :: PR: #12016
- ci: Modular unit tests by @ko3n1g :: PR: #12104
- ci: Update bump workflow by @ko3n1g :: PR: #12106
- etp docs by @akoumpa :: PR: #12111
- build: Better caching by @ko3n1g :: PR: #12109
- ci: Fix flaky test by @ko3n1g :: PR: #12113
- Ensure nemo.collections.vlm does not strictly require transformer engine by @chtruong814 :: PR: #12108
- build: Optimize by @ko3n1g :: PR: #12112
- refactor peft module matching; introduce exclude_modules by @akoumpa :: PR: #12066
- Update mcore commit (02.06.25) by @pablo-garay :: PR: #12114
- ci: Bump Mcore inplace by @ko3n1g :: PR: #12115
- ci: Bump bot by @ko3n1g :: PR: #12117
- Add neva pretrain script by @yaoyu-33 :: PR: #12033
- DAPT playbooks - with NeMo 2.0  by @jvamaraju :: PR: #12067
- Malay/bw scripts by @malay-nagda :: PR: #11961
- [MoE] Add type annotation for mixtral configs by @gdengk :: PR: #12126
- ci: Disable checks by @ko3n1g :: PR: #12129
- Add performance-optimized example for llama2 70b LoRA by @vysarge :: PR: #12055
- Add Automodel support for Deepseek v3 model by @BoxiangW :: PR: #12099
- Bug fix with generation of expert_tensor_parallel_rank by @guyueh1 :: PR: #12125
- Rename neva datamodule by @yaoyu-33 :: PR: #12121
- Update vLLM to 0.7.2 by @Laplasjan107 :: PR: #12078
- Prevent downloading dataset every time in ci test by @cuichenx :: PR: #12095
- AudioToAudioModel: fix model->dataloader sample_rate parameter injection by @racoiaws :: PR: #12092
- Minor Bug Fixes - LLaMa Embedding by @soluwalana :: PR: #12146
- build: Force re-install VCS dependencies by @ko3n1g :: PR: #12155
- Cherry pick `build: Force re-install VCS dependencies (12155)` into `r2.2.0` by @ko3n1g :: PR: #12191
- Cherry pick `Add function calling SFT NeMo2.0 tutorial (11868)` into `r2.2.0` by @ko3n1g :: PR: #12180
- Cherry pick `Update TTS code to remove calls to deprecated functions (12153)` into `r2.2.0` by @ko3n1g :: PR: #12201
- Cherry pick `Fix multi-GPU in-framework deployment (12090)` into `r2.2.0` by @ko3n1g :: PR: #12172
- Cherry pick `disable moe logging to avoid deepseek hang (12168)` into `r2.2.0` by @ko3n1g :: PR: #12192
- Cherry pick `build: Pin down transformers (12229)` into `r2.2.0` by @ko3n1g :: PR: #12230
- Cherry pick `Fix loading extra states from torch tensor (12185)` into `r2.2.0` by @ko3n1g :: PR: #12226
- Cherry pick `nemo-automodel checkpoint-io refactor (12070)` into `r2.2.0` by @ko3n1g :: PR: #12234
- ci: Flaky tests release by @ko3n1g :: PR: #12293
- Cherry pick `Set L2_Speech_Batch_Size_OOMptimizer_Canary to be optional (12299)` into `r2.2.0` by @ko3n1g :: PR: #12300
- build: Editable nemo install (#12304) by @ko3n1g :: PR: #12308
- ci: Fix test workflow by @ko3n1g :: PR: #12311
- Cherry pick `build: Exclude tensorstore 0.1.72 (12317)` into `r2.2.0` by @ko3n1g :: PR: #12318
- Cherry pick `Fix the local path in Sortformer diarizer training tutorial (12135)` into `r2.2.0` by @ko3n1g :: PR: #12316
- Cherry pick `Add eval requirement to setup.py (12152)` into `r2.2.0` by @ko3n1g :: PR: #12277
- Cherry pick `Add modelopt to requirements_nlp.txt (12261)` into `r2.2.0` by @ko3n1g :: PR: #12278
- cherry pick 12209 by @akoumpa :: PR: #12240
- Cherry pick `Energon ckpt multimodal (12245)` into `r2.2.0` by @ko3n1g :: PR: #12307
- Cherry pick `[nemo1] Fix Mamba/Bert loading from checkpoint after TE extra states were introduced (12275)` into `r2.2.0` by @ko3n1g :: PR: #12314
- Cherry pick `fix masked loss calculation (12255)` into `r2.2.0` by @ko3n1g :: PR: #12286
- chore: Cherry pick deepseek by @ko3n1g :: PR: #12324
- build: Bump PyT to 25.01 (#11973) by @ko3n1g :: PR: #12323
- Cherry pick `build: Bump mcore (12320)` into `r2.2.0` by @ko3n1g :: PR: #12328
- Cherry pick `[automodel] re-enable FSDP2 tests (12325)` into `r2.2.0` by @ko3n1g :: PR: #12331
- Cherry pick `[automodel] fix loss reporting (12303)` into `r2.2.0` by @ko3n1g :: PR: #12334
- build: Bump Mcore by @ko3n1g :: PR: #12340
- Cherry-pick Asr fixes 2.2 (#12227) by @ko3n1g :: PR: #12345
- Cherry-pick Bug fixes (#12315) by @chtruong814 :: PR: #12346
- Cherry pick `[automodel] remove fix_progress_bar from fsdp2 strategy (12339)` into `r2.2.0` by @ko3n1g :: PR: #12347
- Cherry pick `Fix NeMo1 Bert Embedding Dataset args (12341)` into `r2.2.0` by @ko3n1g :: PR: #12349
- Cherry pick `Fix NeMo1 sequence_len_offset in Bert fwd (12350)` into `r2.2.0` by @ko3n1g :: PR: #12359
- Cherry pick `Add nemo-run recipe for evaluation (12301)` into `r2.2.0` by @ko3n1g :: PR: #12352
- Cherry pick `Add DeepSeek-R1 Distillation NeMo 2.0 tutorial (12187)` into `r2.2.0` by @ko3n1g :: PR: #12355
- chore: Update package_info.py by @ko3n1g :: PR: #12362
- Version bump to `2.2.0rc4.dev0` by @github-actions[bot] :: PR: #12363
- Bump mcore to latest commit on release branch by @chtruong814 :: PR: #12360
- Cherry pick `[automodel] add lr scheduler (12351)` into `r2.2.0` by @ko3n1g :: PR: #12361
- Cherry pick `[automodel] add distributed data sampler (12326)` into `r2.2.0` by @ko3n1g :: PR: #12373
- Cherry pick `[NeVA] Fix for CP+THD (12366)` into `r2.2.0` by @ko3n1g :: PR: #12375
- Cherry pick `Ignore attribute error when serializing mcore specs (12353)` into `r2.2.0` by @ko3n1g :: PR: #12383
- Cherry pick `Avoid init_ddp for inference (12011)` into `r2.2.0` by @ko3n1g :: PR: #12385
- Cherry pick `[docs] fix notebook render (12374)` into `r2.2.0` by @ko3n1g :: PR: #12394
- Cherry pick `Neva finetune scripts and PP fix (12387)` into `r2.2.0` by @ko3n1g :: PR: #12397
- Cherry pick `[automodel] update runner tags for notebooks (12428)` into `r2.2.0` by @ko3n1g :: PR: #12431
- Cherry pick `[automodel] update examples (12411)` into `r2.2.0` by @ko3n1g :: PR: #12432
- Cherry pick `Evaluation docs (12348)` into `r2.2.0` by @ko3n1g :: PR: #12460
- Cherry pick `Update prompt format (12452)` into `r2.2.0` by @ko3n1g :: PR: #12455
- Cherry pick `Fixing a wrong Sortformer Tutorial Notebook path. (12479)` into `r2.2.0` by @ko3n1g :: PR: #12480
- Cherry pick `added a needed checks and changes for bugfix (12400)` into  `r2.2.0` by @Ssofja :: PR: #12447
- Cherry pick `[automodel] fix loss/tps reporting across ranks (12389)` into `r2.2.0` by @ko3n1g :: PR: #12413
- Cherry pick `enable fsdp flag for FSDP2Strategy (12392)` into `r2.2.0` by @ko3n1g :: PR: #12429
- Cherry pick `Fix lita notebook issue (12474)` into `r2.2.0` by @ko3n1g :: PR: #12476
- Cherrypick multinode tut changes by @BoxiangW :: PR: #12501
- Cherry pick `Changed the argument types passed to metrics calculation functions (12500)` into `r2.2.0` by @ko3n1g :: PR: #12502
- Cherry pick `added needed fixes (12495)` into `r2.2.0` by @ko3n1g :: PR: #12509
- Cherry pick `update transformers version requirements (12475)` into `r2.2.0` by @ko3n1g :: PR: #12507
- Cherry pick `[checkpoint] Log timings for checkpoint IO save and load (11972)` into `r2.2.0` by @ko3n1g :: PR: #12520
- Cherry pick `few checkings needed because of the change of asr models output (12499)` into `r2.2.0` by @ko3n1g :: PR: #12513
- Oyilmaz nvidia/chore/cherry pick 12242 by @oyilmaz-nvidia :: PR: #12523
- Cherry pick `Remove`_attn_implementation` in `LlamaBidirectionalModel`constructor (12364)` into `r2.2.0` by @ko3n1g :: PR: #12525
- Cherry pick `Configure FSDP to keep module params (12074)` into `r2.2.0` by @ko3n1g :: PR: #12524
- Cherry pick `[automodel] docs (11942)` into `r2.2.0` by @ko3n1g :: PR: #12530
- Cherry pick `[automodel] update examples' comments (12518)` and `[automodel] Move PEFT to configure_model (#12491)` into `r2.2.0` by @ko3n1g :: PR: #12529
- Cherry pick `update readme to include latest pytorch version (12539)` into `r2.2.0` by @ko3n1g :: PR: #12577
- Publish r2.2.0 by @chtruong814 :: PR: #12583

</details>

## NVIDIA Neural Modules 2.1.0

### Highlights

- Training
  - Fault Tolerance
    - Straggler Detection
    - Auto Relaunch
- LLM & MM
  - MM models
    - Llava-next
    - Llama 3.2
  - Sequence Model Parallel for NeVa
  - Enable Energon
  - SigLIP (NeMo 1.0 only)
  - LLM 2.0 migration
    - Starcoder2
    - Gemma 2
    - T5
    - Baichuan
    - BERT
    - Mamba
    - ChatGLM
  - DoRA support
- Export
  - Nemo 2.0 base model export path for NIM
  - PTQ in Nemo 2.0
- ASR
  - Timestamps with TDT decoder
  - Timestamps option with .transcribe()

### Detailed Changelogs

#### ASR

<details><summary>Changelog</summary>

- [Fix] Fixed sampler override and audio_key in prepare_audio_data by @anteju :: PR: #10980
- Akoumparouli/mixtral recipe fix r2.0.0 by @akoumpa :: PR: #10994
- TDT compute timestamps option and Extra Whitespace handling for SPE by @monica-sekoyan :: PR: #10875
- ci: Switch to CPU only runner by @ko3n1g :: PR: #11035
- Fix timestamps tests by @monica-sekoyan :: PR: #11053
- ci: Pin release freeze by @ko3n1g :: PR: #11143
- Fix RNN-T loss memory usage by @artbataev :: PR: #11144
- Added deprecation notice by @Ssofja :: PR: #11133
- Fixes for Canary adapters tutorial by @pzelasko :: PR: #11184
- add ipython import guard by @nithinraok :: PR: #11191
- Self Supervised Pre-Training tutorial Fix by @monica-sekoyan :: PR: #11206
- update the return type by @nithinraok :: PR: #11210
- Timestamps to transcribe by @nithinraok :: PR: #10950
- [Doc fixes] update file names, installation instructions, bad links by @erastorgueva-nv :: PR: #11045
- Beam search algorithm implementation for TDT models by @lilithgrigoryan :: PR: #10903
- Update import 'pytorch_lightning' -> 'lightning.pytorch' by @maanug-nv :: PR: #11252
- Remove pytorch-lightning by @maanug-nv :: PR: #11306
- update hypothesis when passed through cfg by @nithinraok :: PR: #11366
- Revert "update hypothesis when passed through cfg" by @pablo-garay :: PR: #11373
- Fix transcribe speech by @nithinraok :: PR: #11379
- Lhotse support for transcribe_speech_parallel by @nune-tadevosyan :: PR: #11249
- Sortformer Diarizer 4spk v1 model PR Part 1: models, modules and dataloaders by @tango4j :: PR: #11282
- Removing unnecessary lines by @nune-tadevosyan :: PR: #11408
- Support for initializing lhotse shar dataloader via field: list[path] mapping by @pzelasko :: PR: #11460
- New extended prompt format for Canary, short utterances inference fix, and training micro-optimizations by @pzelasko :: PR: #11058
- Fixing Multi_Task_Adapters.ipynb by replacing canary2 with canary_custom by @weiqingw4ng :: PR: #11636

</details>

#### TTS

<details><summary>Changelog</summary>

- [Doc fixes] update file names, installation instructions, bad links by @erastorgueva-nv :: PR: #11045
- Add T5TTS by @blisc :: PR: #11193
- Update import 'pytorch_lightning' -> 'lightning.pytorch' by @maanug-nv :: PR: #11252
- Remove pytorch-lightning by @maanug-nv :: PR: #11306
- Add nvidia/low-frame-rate-speech-codec-22khz model on docs by @Edresson :: PR: #11457

</details>

#### NLP / NMT

<details><summary>Changelog</summary>

- Move collectiob.nlp imports inline for t5 by @marcromeyn :: PR: #10877
- Use a context-manager when opening files by @akoumpa :: PR: #10895
- Packed sequence bug fixes by @cuichenx :: PR: #10898
- ckpt convert bug fixes by @dimapihtar :: PR: #10878
- remove deprecated ci tests by @dimapihtar :: PR: #10922
- Update T5 tokenizer (adding additional tokens to tokenizer config) by @huvunvidia :: PR: #10972
- Add support and recipes for HF models via AutoModelForCausalLM by @akoumpa :: PR: #10962
- gpt3 175b cli by @malay-nagda :: PR: #10985
- Fix for crash with LoRA + tp_overlap_comm=false + sequence_parallel=true by @vysarge :: PR: #10920
- Update `BaseMegatronSampler` for compatibility with PTL's `_BatchProgress` by @ashors1 :: PR: #11016
- add deprecation note by @dimapihtar :: PR: #11024
- Update ModelOpt Width Pruning example defaults by @kevalmorabia97 :: PR: #10902
- switch to NeMo 2.0 recipes by @dimapihtar :: PR: #10948
- NeMo 1.0: upcycle dense to moe by @akoumpa :: PR: #11002
- Gemma2 in Nemo2 with Recipes by @suiyoubi :: PR: #11037
- Add Packed Seq option to GPT based models by @suiyoubi :: PR: #11100
- Fix MCoreGPTModel import in llm.gpt.model.base by @hemildesai :: PR: #11109
- TP+MoE peft fix by @akoumpa :: PR: #11114
- GPT recipes to use full te spec by @JimmyZhang12 :: PR: #11119
- Virtual pipeline parallel support for LoRA in NLPAdapterModelMixin by @vysarge :: PR: #11128
- update nemo args for mcore flash decode arg change by @HuiyingLi :: PR: #11138
- Call `ckpt_to_weights_subdir` from `MegatronCheckpointIO` by @ashors1 :: PR: #10897
- [Doc fixes] update file names, installation instructions, bad links by @erastorgueva-nv :: PR: #11045
- fix(export): GPT models w/ bias=False convert properly by @terrykong :: PR: #11255
- Use MegatronDataSampler in HfDatasetDataModule by @akoumpa :: PR: #11274
- Add T5TTS by @blisc :: PR: #11193
- ci: Exclude CPU machines from scan by @ko3n1g :: PR: #11300
- Revert "fix(export): GPT models w/ bias=False convert properly" by @terrykong :: PR: #11301
- remove redundant docs by @sharathts :: PR: #11302
- Update import 'pytorch_lightning' -> 'lightning.pytorch' by @maanug-nv :: PR: #11252
- Add `attention_bias` argument in transformer block and transformer layer modules, addressing change in MCore by @yaoyu-33 :: PR: #11289
- Remove pytorch-lightning by @maanug-nv :: PR: #11306
- Update T5 attention-mask shapes to be compatible with all attention-backend in new TE versions by @huvunvidia :: PR: #11059
- Add support for restoring from 2.0 checkpoint in 1.0 by @hemildesai :: PR: #11347
- Fix Gemma2 Attention Args by @suiyoubi :: PR: #11365
- mlm conversion & tiktokenizer support by @dimapihtar :: PR: #11349
- [Nemo1] Generate sharded optimizer state dicts only if needed for saving by @ananthsub :: PR: #11451
- add hindi tn/itn coverage by @mgrafu :: PR: #11382
- chore(beep boop 🤖): Bump `MCORE_TAG=67a50f2...` (2024-11-28) by @ko3n1g :: PR: #11427
- Handle exception when importing RetroGPTChunkDatasets by @guyueh1 :: PR: #11415
- Update restore from config for gpt type continual training in NeMo1 by @yaoyu-33 :: PR: #11471
- ci: Re-enable `L2_Megatron_LM_To_NeMo_Conversion` by @ko3n1g :: PR: #11484
- Apply packed sequence params change for fused rope compatibility by @ananthsub :: PR: #11506
- Huvu/tiktoken tokenizer update by @huvunvidia :: PR: #11494

</details>

#### Text Normalization / Inverse Text Normalization

<details><summary>Changelog</summary>

- Adding support for LightningDataModule inside Fabric-API by @marcromeyn :: PR: #10879
- Add registry to register all needed classes with artifacts in nemo.lightning.io by @hemildesai :: PR: #10861
- Update import 'pytorch_lightning' -> 'lightning.pytorch' by @maanug-nv :: PR: #11252
- Remove pytorch-lightning by @maanug-nv :: PR: #11306
- add hindi tn/itn coverage by @mgrafu :: PR: #11382

</details>

#### Export

<details><summary>Changelog</summary>

- Update engine build step for TRT-LLM 0.13.0 by @janekl :: PR: #10880
- Nemo 2.0 ckpt support in TRT-LLM export by @oyilmaz-nvidia :: PR: #10891
- Fix TRTLLM parallel_embedding by @meatybobby :: PR: #10975
- Export & deploy updates (part I) by @janekl :: PR: #10941
- Add doc-strings to import & export + improve logging by @marcromeyn :: PR: #11078
- NeMo-UX: fix nemo-ux export path by @akoumpa :: PR: #11081
- Fix TRTLLM nemo2 activation parsing by @meatybobby :: PR: #11062
- Support exporting Nemotron-340B for TensorRT-LLM by @jinyangyuan-nvidia :: PR: #11015
- vLLM Hugging Face exporter by @oyilmaz-nvidia :: PR: #11124
- Fix export of configuration parameters to Weights and Biases by @soluwalana :: PR: #10995
- Change activation parsing in TRTLLM by @meatybobby :: PR: #11173
- Remove builder_opt param from trtllm-build for TensorRT-LLM >= 0.14.0 by @janekl :: PR: #11259
- fix(export): GPT models w/ bias=False convert properly by @terrykong :: PR: #11255
- fix(export): update API for disabling device reassignment in TRTLLM for Aligner by @terrykong :: PR: #10863
- Add openai-gelu in gated activation for TRTLLM export by @meatybobby :: PR: #11293
- Revert "fix(export): GPT models w/ bias=False convert properly" by @terrykong :: PR: #11301
- Adding alinger export by @shanmugamr1992 :: PR: #11269
- Export & deploy updates (part II) by @janekl :: PR: #11344
- Introducing TensorRT lazy export and caching option with trt_compile()  by @borisfom :: PR: #11266
- fix: export converts properly if no model_prefix by @terrykong :: PR: #11477

</details>

#### Bugfixes

<details><summary>Changelog</summary>

- Change default ckpt name by @maanug-nv :: PR: #11277
- Fix patching of NeMo tokenizers for correct Lambada evaluation by @janekl :: PR: #11326

</details>

#### Uncategorized

<details><summary>Changelog</summary>

- ci: Use Slack group by @ko3n1g :: PR: #10866
- Bump `Dockerfile.ci` (2024-10-14) by @ko3n1g :: PR: #10871
- Fix peft resume by @cuichenx :: PR: #10887
- call __post_init__ after altering config values by @akoumpa :: PR: #10885
- Late import prettytable by @maanug-nv :: PR: #10912
- Bump `Dockerfile.ci` (2024-10-17) by @ko3n1g :: PR: #10919
- Warning for missing FP8 checkpoint support for vLLM deployment by @janekl :: PR: #10906
- Fix artifact saving by @hemildesai :: PR: #10914
- Lora improvement by @cuichenx :: PR: #10918
- Huvu/t5 nemo2.0 peft by @huvunvidia :: PR: #10916
- perf recipes and Mcore DistOpt params by @malay-nagda :: PR: #10883
- ci: Fix cherry pick team by @ko3n1g :: PR: #10945
- Fix requirements for MacOS by @artbataev :: PR: #10930
- Fix nemo 2.0 recipes  by @BoxiangW :: PR: #10915
- Akoumparouli/nemo ux fix dir or string artifact by @akoumpa :: PR: #10936
- Fix typo in docstring by @ashors1 :: PR: #10955
- [Nemo CICD] Remove deprecated tests by @pablo-garay :: PR: #10960
- Restore NeMo 2.0 T5 pretraining CICD test by @huvunvidia :: PR: #10952
- Convert perf plugin env vars to strings by @hemildesai :: PR: #10947
- disable dynamo for ddp checker by @akoumpa :: PR: #10961
- Bump `Dockerfile.ci` (2024-10-21) by @ko3n1g :: PR: #10965
- respect warnings' filters by @akoumpa :: PR: #10953
- Alit/mamba recipe by @JRD971000 :: PR: #10935
- Long context performance doc hot fix by @youngeunkwon0405 :: PR: #10946
- Performance mode by @malay-nagda :: PR: #10926
- Bump `Dockerfile.ci` (2024-10-22) by @ko3n1g :: PR: #10979
- Add more recipes by @cuichenx :: PR: #10957
- ci: Update tests by @ko3n1g :: PR: #10987
- Bump `Dockerfile.ci` (2024-10-23) by @ko3n1g :: PR: #11001
- llm.generate fixes by @HuiyingLi :: PR: #10983
- use __dict__ in check by @akoumpa :: PR: #11012
- LoRA support for HF::AutoModelForCausalLM by @akoumpa :: PR: #10982
- Change default for always_save_context to True by @athitten :: PR: #11014
- Fix pip install by @marcromeyn :: PR: #11026
- Change dist ckpt defaults by @ShriyaPalsamudram :: PR: #10913
- Fix _strategy_lib tests by @maanug-nv :: PR: #11033
- Basic online dynamic FP8 quantization with vLLM by @janekl :: PR: #10904
- Expose packed seq in finetuning recipes by @cuichenx :: PR: #11006
- PEFT Inference by @cuichenx :: PR: #11030
- added Lhotse online augmentation tutorial for SE by @nasretdinovr :: PR: #10944
- Bump `Dockerfile.ci` (2024-10-27) by @ko3n1g :: PR: #11051
- ci: Send team alerts on specific keywords by @ko3n1g :: PR: #10986
- Qwen2 Recipe by @suiyoubi :: PR: #10974
- Bump `Dockerfile.ci` (2024-10-28) by @ko3n1g :: PR: #11054
- Generalizing Inference pipeline in NeMo 2.0 to support encoder-decoder models by @huvunvidia :: PR: #10924
- [Bug fix] In energon MultiModalSampleConfig use default_factory in dataclass by @guyueh1 :: PR: #11041
- fix: Resolve mutable default issue in MultiModalSampleConfig dataclass by @michal2409 :: PR: #11061
- SC1/SC2 Recipe by @suiyoubi :: PR: #10971
- Wrap batch_sampler with_IndexBatchSamplerWrapper by @farhadrgh :: PR: #10934
- Performance fine-tuning recipes for llama3 8b + 70b by @vysarge :: PR: #11046
- Set TE spec name for NeMo to HF checkpoint converters by @kevalmorabia97 :: PR: #11036
- ci: Re-add secrets detector by @ko3n1g :: PR: #11038
- Adding nemo-run recipes for NeMo 2.0 T5  by @huvunvidia :: PR: #10964
- Minor fixes for NeMo 2.0 PTQ by @Laplasjan107 :: PR: #11079
- Add copyright check by @pablo-garay :: PR: #11048
- Fix finalize model grad for PEFT by @cuichenx :: PR: #11065
- ci: Less verbose infra alerts by @ko3n1g :: PR: #11080
- Add copyright notice by @pablo-garay :: PR: #11085
- ci: Fix cron schedule  by @ko3n1g :: PR: #11076
- ci: Use code-freeze via Nemo-FW-Templates by @ko3n1g :: PR: #11073
- Akoumparouli/hf lit module peft ckpt bugfix by @akoumpa :: PR: #11022
- PEFT perf and TE spec fixes by @JimmyZhang12 :: PR: #11070
- Bump `Dockerfile.ci` (2024-10-30) by @ko3n1g :: PR: #11092
- NeMorun for NeMo 2.0 T5 finetuning by @huvunvidia :: PR: #11040
- fix model_checkpoint.py by @ethanhe42 :: PR: #11057
- Update PTQ tests and ModelOpt version by @janekl :: PR: #11095
- Fix datasets in CLI by @marcromeyn :: PR: #11097
- Fix yaml serialization in io mixin by @hemildesai :: PR: #11106
- disable overlap_param_gather_with_optimizer_step by @JimmyZhang12 :: PR: #11102
- nemo1 to nemo2 checkpoint convert by @HuiyingLi :: PR: #10937
- fix expert regex filter by @akoumpa :: PR: #11103
- Remove stale checkpoint deletion on checkpoint saving failure by @akoumpa :: PR: #11116
- NeMo-UX: Mistral/mixtral peft ci test by @akoumpa :: PR: #11094
- Make nemo.collections.llm PreTrainingDataModule num samples configurable by @hemildesai :: PR: #11088
- Fix packed seq path by @cuichenx :: PR: #11121
- Allow arguments passed to dataset class + Gemma recipe fix by @cuichenx :: PR: #11125
- Nemotron Recipe by @suiyoubi :: PR: #11118
- NeMo-UX: HF PeFT fix by @akoumpa :: PR: #11096
- Remove deprecated tests by @pablo-garay :: PR: #11134
- Recipe Fix for NeMo CI by @suiyoubi :: PR: #11127
- Fix freeze_model call in peft by @cuichenx :: PR: #11146
- Bump `Dockerfile.ci` (2024-11-05) by @ko3n1g :: PR: #11159
- NeMo-UX: Add sgd optim by @akoumpa :: PR: #11157
- Update copyright check by @pablo-garay :: PR: #11168
- add lora recipt for 405b by @JRD971000 :: PR: #10991
- dit training diagrams by @zpx01 :: PR: #10873
- ci: Switch to FW templates for build by @ko3n1g :: PR: #11077
- Bump `Dockerfile.ci` (2024-11-06) by @ko3n1g :: PR: #11174
- feat: Run PyLint by @ko3n1g :: PR: #11147
- Add Alpaca Finetune Datamodule by @suiyoubi :: PR: #11185
- Updated Diffusion Collection README by @zpx01 :: PR: #11179
- Add support for Cosmos Tokenizers by @jojennin :: PR: #11194
- Run formatting only if files changed. Echo message if pylint fails. by @artbataev :: PR: #11188
- Bump `Dockerfile.ci` (2024-11-07) by @ko3n1g :: PR: #11196
- Fix rotary_percentage parsing in nemo2 config by @meatybobby :: PR: #11197
- ci: Update cherry pick workflow by @ko3n1g :: PR: #11202
- ci: Build, test, publish a wheel by @ko3n1g :: PR: #11183
- Bump `Dockerfile.ci` (2024-11-08) by @ko3n1g :: PR: #11222
- update default pipeline_parallelism_type by @akoumpa :: PR: #11213
- check actual value of vocab_file by @akoumpa :: PR: #11228
- Fix VP Initialization Issue with Latest MCore by @suiyoubi :: PR: #11209
- ci: Run Pylint strictly on new files, softly on history by @ko3n1g :: PR: #11212
- ci: Add release workflow by @ko3n1g :: PR: #11180
- Fix llm.generate by @hemildesai :: PR: #11217
- Bump `Dockerfile.ci` (2024-11-11) by @ko3n1g :: PR: #11247
- Bump `Dockerfile.ci` (2024-11-12) by @ko3n1g :: PR: #11254
- Handling tokenizer in PTQ for Nemo 2.0 by @janekl :: PR: #11237
- Fix finetuning datamodule resume by @cuichenx :: PR: #11187
- ci: Move `bump mcore` to templates by @ko3n1g :: PR: #11229
- ci: Fix secrets detector by @ko3n1g :: PR: #11205
- chore(beep boop 🤖): Bump `MCORE_TAG=aded519...` (2024-11-12) by @ko3n1g :: PR: #11260
- ci: Run secrets detector on `pull_request_target` by @ko3n1g :: PR: #11263
- Advanced Diffusion Training Features by @zpx01 :: PR: #11246
- Update pruning and distillation tutorial notebooks by @gvenkatakris :: PR: #11091
- update nemo1->2 conversion according to changes in main by @HuiyingLi :: PR: #11253
- Add llama 3.1 recipes by @cuichenx :: PR: #11273
- Fix Finetune Recipe by @suiyoubi :: PR: #11267
- Configure no restart validation loop in nl.Trainer by @hemildesai :: PR: #11029
- Handle _io_unflatten_object when_thread_local.output_dir is not available by @hemildesai :: PR: #11199
- Remove opencc upperbound by @thomasdhc :: PR: #10909
- Fix head_size in NeMo to HF checkpoint converters for width pruned model support by @eagle705 :: PR: #11230
- Fixes per comments by @gvenkatakris :: PR: #11280
- Create phi3mini.py by @mayani-nv :: PR: #11281
- ci: Fix release workflow by @ko3n1g :: PR: #11286
- fix perf plugin CUDA_DEVICE_MAX_CONNECTIONS setting by @JimmyZhang12 :: PR: #11299
- PTQ via NeMo-Run CLI by @janekl :: PR: #10984
- PTQ memory optimization by @Laplasjan107 :: PR: #11257
- Update README.md for collection page by @yaoyu-33 :: PR: #11223
- Adding multimodal examples by @shanmugamr1992 :: PR: #11279
- Add HF untrusted code toggle by @akoumpa :: PR: #11313
- P2p chunk size setting in nemo 2.0 by @erhoo82 :: PR: #11312
- Nemo2 batcheval by @HuiyingLi :: PR: #11158
- DoRA by @cuichenx :: PR: #11104
- Profiling - support Chakra & Kineto trace dumping by @lilyw97 :: PR: #11115
- NeMo 2.0 SFT PEFT notebooks by @HuiyingLi :: PR: #10874
- Update symlink option for save_last in ModelCheckpoint by @paul-gibbons :: PR: #11319
- ci: Pass-through of `workflow_event` by @ko3n1g :: PR: #11322
- Add StragglerDetection and auto-relaunch to NeMo2.0 by @ShriyaPalsamudram :: PR: #11328
- Huvu/t5 nemo2.0 nemoci by @huvunvidia :: PR: #11291
- TE acceleration using callbacks by @oyilmaz-nvidia :: PR: #11261
- Leave target_module as default in PEFT Recipes by @cuichenx :: PR: #11334
- More robust tar file loading from AIStore by @pzelasko :: PR: #11323
- Fix CLIP transformer layer api by @yaoyu-33 :: PR: #11337
- pass trust_remote_code to AutoTokenizer by @akoumpa :: PR: #11343
- Fix linear layer replacement by @oyilmaz-nvidia :: PR: #11356
- fix typo by @JRD971000 :: PR: #11351
- Add torchrun local executor to recipes by @marcromeyn :: PR: #11342
- Add PP support in NeVA along with few bug fixes by @yaoyu-33 :: PR: #11170
- nemo2 peft merge by @HuiyingLi :: PR: #11017
- Add dora recipes by @cuichenx :: PR: #11330
- add fix to recipe by @JRD971000 :: PR: #11368
- Add missing test to CICD needed list by @pablo-garay :: PR: #11376
- update SquadDataModule to use run.config by @huvunvidia :: PR: #11358
- Add llama 3.2 1b and 3b by @cuichenx :: PR: #11335
- calculate metrics for nemo2 sftpeft notebook by @HuiyingLi :: PR: #11381
- Enable packed dataset for validation; add a2a_experimental argument by @michal2409 :: PR: #11378
- Fix DDP unused param error when TE is enabled in NeMo Lite by @oyilmaz-nvidia :: PR: #11364
- Update llama32 vision (mllama) use attention bias by @yaoyu-33 :: PR: #11316
- Fix environment variables in torchrun executor by @hemildesai :: PR: #11363
- Add sample generate to PTQ for NeMo 2.0 by @Laplasjan107 :: PR: #11339
- Fix selective restore by explicitly verifying keys by @hemildesai :: PR: #11377
- Minor fix by @gvenkatakris :: PR: #11353
- Add a fix for single-GPU nsys. by @tfogal :: PR: #11354
- capitalize HF as HF instead of Hf by @akoumpa :: PR: #11384
- ci: Add HF cache by @ko3n1g :: PR: #11398
- Remove logic to skip checkpoint save if checkpoint exists by @ashors1 :: PR: #11362
- Rewire tokenizer exception handling in model resume by @cuichenx :: PR: #11375
- Adding LLava-Next model class by @yashaswikarnati :: PR: #11399
- Fix vllm test issue when run_accuracy is enabled by @oyilmaz-nvidia :: PR: #11413
- data modules for llava_next by @yashaswikarnati :: PR: #11400
- Fix strategies saving unsharded optimizer states by @ananthsub :: PR: #11392
- Adjust CLI support for PTQ by @janekl :: PR: #11421
- Nemo run recipe's and example scripts for Llava Next by @yashaswikarnati :: PR: #11405
- Huvu/t5 nemo2.0 nemoci 3b11b by @huvunvidia :: PR: #11388
- ci: Allow dry-run of release by @ko3n1g :: PR: #11418
- fix dtype when init HF model from config by @akoumpa :: PR: #11420
- Handle import errors in virtual environment when running vLLM tests by @janekl :: PR: #11435
- Fix loss mask when answer_only_loss=True by @ashors1 :: PR: #11444
- [audio] Keep input directory structure when saving processed files by @anteju :: PR: #11403
- Add different recipe examples to NeMo 2.0 by @BoxiangW :: PR: #11317
- [Scripts] Remove fixed seed for adding noise by @anteju :: PR: #11401
- Add option to provide prior NeMo 2 ckpt path to convert_nemo1_to_nemo… by @hemildesai :: PR: #11452
- PTQ CLI and param updates by @janekl :: PR: #11459
- Add tests for resiliency feature integration by @maanug-nv :: PR: #11406
- ci: Disable HexHighEntropyString plugin by @ko3n1g :: PR: #11470
- Fix broken links by @shashank3959 :: PR: #11294
- Nemo 2.0 canonical lora by @cuichenx :: PR: #11416
- ci: Run secrets detector on merge-commit by @ko3n1g :: PR: #11479
- Formatting (minor) by @pablo-garay :: PR: #11485
- Fix bug related to naming by @pablo-garay :: PR: #11487
- Add BERT Model To NeMo2.0 by @suiyoubi :: PR: #11333
- Update Nemo Distributed Checkpoint User Guide by @FortunaZhang :: PR: #11489
- fix: regular torch optims (e.g., sgd) no longer error with closure spec by @terrykong :: PR: #11189
- Add recipe configs validating by @BoxiangW :: PR: #10954
- Fix finetuning PP by @cuichenx :: PR: #11474
- [docs] Documentation for audio collection by @anteju :: PR: #11426
- config hierarchy by @malay-nagda :: PR: #11145
- Force param sync when using distributed optimizer and overlap_param_gather by @hemildesai :: PR: #11486
- chore(beep boop 🤖): Bump `MCORE_TAG=bd677bf...` (2024-12-06) by @ko3n1g :: PR: #11492
- Remove default mutable arguments from AbstractEmbModel constructor by @ananthsub :: PR: #11348
- minor fix for nemo2 sftpeft readme by @HuiyingLi :: PR: #11502
- Update Llama3 Fine-Tuning Notebook by @roclark :: PR: #11522
- Fix CI issue on validation config by @BoxiangW :: PR: #11521
- Freeze tags in in `r2.1.0` by @github-actions[bot] :: PR: #11556
- Cherrypick all + R2.1.0 fix cicd  by @pablo-garay :: PR: #11622
- Cherry pick `Add fix docstring for speech commands (11638)` into `r2.1.0` by @ko3n1g :: PR: #11639
- Cherrypick #11628 to r2.1.0 by @nasretdinovr :: PR: #11630
- Update package_info.py by @ko3n1g :: PR: #11646
- Cherry pick `Add fix docstring for VAD (11659)` into `r2.1.0` by @ko3n1g :: PR: #11660
- Fix tokenizer trust_remote_code by @cuichenx :: PR: #11657
- Cherrypick 11568 by @cuichenx :: PR: #11656
- Cherry pick `Downgrading the 'datasets' package from 3.0.0 to 2.21.0 for Multilang_ASR.ipynb and ASR_CTC_Language_Finetuning.ipynb (11675)` into `r2.1.0` by @ko3n1g :: PR: #11677
- r2.1.0 cherrypick by @pablo-garay :: PR: #11680
- Cherry pick `Rename multimodal data module - EnergonMultiModalDataModule (11654)` into `r2.1.0` by @ko3n1g :: PR: #11685
- chore: Bump to `r2.1.0rc2` by @ko3n1g :: PR: #11693
- r2.1.0 ptl fix by @pablo-garay :: PR: #11694

</details>

## NVIDIA Neural Modules 2.1.0rc2

Prerelease: NVIDIA Neural Modules 2.1.0rc2 (2024-12-21)

## NVIDIA Neural Modules 2.1.0rc1

Prerelease: NVIDIA Neural Modules 2.1.0rc1 (2024-12-20)

## NVIDIA Neural Modules 2.1.0rc0

Prerelease: NVIDIA Neural Modules 2.1.0rc0 (2024-12-12)

## NVIDIA Neural Modules 2.0.0rc1

### Highlights

#### Large language models

- PEFT: QLoRA support, LoRA/QLora for Mixture-of-Experts (MoE) dense layer
- State Space Models & Hybrid Architecture support (Mamba2 and NV-Mamba2-hybrid)
- Support Nemotron, Minitron, Gemma2, Qwen, RAG
- Custom Tokenizer training in NeMo
- Update the Auto-Configurator for EP, CP and FSDP

#### Multimodal

- NeVA: Add SOTA LLM backbone support (Mixtral/LLaMA3) and suite of model parallelism support (PP/EP)
- Support Language Instructed Temporal-Localization Assistant (LITA) on top of video NeVA

#### ASR

- SpeechLM and SALM
- Adapters for Canary Customization
- Pytorch allocator in PyTorch 2.2 improves training speed up to 30% for all ASR models
- Cuda Graphs for Transducer Inference
- Replaced webdataset with Lhotse - gives up to 2x speedup
- Transcription Improvements - Speedup and QoL Changes
- ASR Prompt Formatter for multimodal Canary

#### Export & Deploy

- In framework PyTriton deployment with backends: - PyTorch - vLLM - TRT-LLM update to 0.10
- TRT-LLM C++ runtime

### Detailed Changelogs

#### ASR

<details><summary>Changelog</summary>

- Support dataloader as input to `audio` for transcription by @titu1994 :: PR: #9201
- Clean up dev docs collection section by @yaoyu-33 :: PR: #9205
- Fix Online_Offline_Microphone_VAD_Demo.ipynb by @stevehuang52 :: PR: #9251
- Remove .nemo instead of renaming by @mikolajblaz :: PR: #9281
- Fix GreedyBatchedCTCInfer regression from GreedyCTCInfer. by @galv :: PR: #9347
- Revert "Fix GreedyBatchedCTCInfer regression from GreedyCTCInfer." by @titu1994 :: PR: #9351
- Prompt formatter API and canary transcribe tensor input support by @pzelasko :: PR: #9206
- Fix prompt formatter's defaults=None case in multi-task model by @pzelasko :: PR: #9366
- move AED chunked infer script by @stevehuang52 :: PR: #9367
- Use model-cast-to-bfloat16 rather than AMP-to-bfloat16 for inference. by @galv :: PR: #9198
- ci: Fix `L2_Segmentation_Tool_Parallel_ctc_segmentation_test_L2_Eng_C… by @ko3n1g :: PR: #9399
- Fix logging message for ASR by @titu1994 :: PR: #9469
- Add support to change Multi task model prompt by @titu1994 :: PR: #9542
- Enable encoder adapters for Canary and MultiTaskAED models by @titu1994 :: PR: #9409
- Audio model collection by @anteju :: PR: #9263
- TitaNet Batch Verify Speaker by @monica-sekoyan :: PR: #9337
- Fix the arguments  of forward_for_export function in msdd_models by @tango4j :: PR: #9624
- chore: Pin branch in notebooks by @ko3n1g :: PR: #9697
- refactor: notebook branch release by @ko3n1g :: PR: #9711
- Canary Adapters tutorial (#9670) by @nithinraok :: PR: #9777
- typos and branch name update to r2.0.0rc1 by @nithinraok :: PR: #9846
- Fix RNNT alignments test by @artbataev :: PR: #9770
- By default trust remote code from HF Datasets by @nithinraok :: PR: #9886
- Temporarily disable cuda graph based RNN-T greedy inference for r2.0.0rc1 by @galv :: PR: #9904
- Enable CUDA graphs by default, but require CUDA 12.6 for full graphs by @artbataev :: PR: #9919
- update branch name for script by @nithinraok :: PR: #9936
- updte branch by @nithinraok :: PR: #9942

</details>

#### TTS

<details><summary>Changelog</summary>

- Clean up dev docs collection section by @yaoyu-33 :: PR: #9205
- Add mel codec checkpoints by @anteju :: PR: #9228
- GPU unit tests: Mark flaky tests to be fixed by @pablo-garay :: PR: #9559
- chore: Pin branch in notebooks by @ko3n1g :: PR: #9697
- refactor: notebook branch release by @ko3n1g :: PR: #9711

</details>

#### LLM/Multimodal

<details><summary>Changelog</summary>

- Update nemo.export module for quantized models by @janekl :: PR: #9218
- Add save option to the TRT-LLM export test script by @oyilmaz-nvidia :: PR: #9221
- Checkpoint resuming compatible for 2403 container by @suiyoubi :: PR: #9199
- Clean up dev docs collection section by @yaoyu-33 :: PR: #9205
- use get with fallback when reading checkpoint_callback_params by @akoumpa :: PR: #9223
- Revert rope fusion defaults by @cuichenx :: PR: #9237
- fix import by @akoumpa :: PR: #9240
- Add TRT-LLM params like max_num_tokens and opt_num_tokens by @oyilmaz-nvidia :: PR: #9210
- sum-reduce grad_norm in DP+CP domain by @erhoo82 :: PR: #9262
- Alit/bert convert fix by @JRD971000 :: PR: #9285
- conv1d stable version by @JRD971000 :: PR: #9330
- Fix trainer builder when exp_manager is not in config by @yaoyu-33 :: PR: #9293
- Fix Peft Weights Loading in NeVA by @yaoyu-33 :: PR: #9341
- Skip sequence_parallel allreduce when using Mcore DistOpt by @akoumpa :: PR: #9344
- Fix FSDP gradient calculation with orig params by @janEbert :: PR: #9335
- TRT-LLM Export Code Cleanup by @oyilmaz-nvidia :: PR: #9270
- support null/None truncation field by @arendu :: PR: #9355
- NeVa token fusion by @paul-gibbons :: PR: #9245
- bugfix if using mcore distOpt with sft by @akoumpa :: PR: #9356
- Re-org export code by @oyilmaz-nvidia :: PR: #9353
- QLoRA by @cuichenx :: PR: #9340
- PeFT fix for distOpt by @akoumpa :: PR: #9392
- [NeMo-UX] Integrating mcore's DistributedDataParallel into MegatronStrategy by @marcromeyn :: PR: #9387
- cherry pick of #9266 by @dimapihtar :: PR: #9411
- Enable specifying alpha for PTQ INT8 SmoothQuant method by @janekl :: PR: #9423
- add support for new mcore ds features by @dimapihtar :: PR: #9388
- LoRA for MoE Layer by @cuichenx :: PR: #9396
- Mistral-7B: apply user's precision to output checkpoint by @akoumpa :: PR: #9222
- Add option to merge distributed optimizer buckets by @timmoon10 :: PR: #9414
- TRT-LLM 0.10 Update by @oyilmaz-nvidia :: PR: #9402
- In-framework deployment by @oyilmaz-nvidia :: PR: #9438
- Bugfix missing variables and argument changes to MegatronPretrainingRandomSampler by @jstjohn :: PR: #9458
- Hyena Operator by @guyjacob :: PR: #9264
- Refactor Quantizer for reusing in QAT by @kevalmorabia97 :: PR: #9276
- move load state dict after initialize parallel state in nlp_model by @ryxli :: PR: #9382
- Enable user to optionally upgrade Megatron by @jstjohn :: PR: #9478
- Fix unwrap model by @cuichenx :: PR: #9480
- fix operator precedence by @akoumpa :: PR: #9403
- [NeMo-UX] Adding context- & expert-parallelism to MegatronStrategy by @marcromeyn :: PR: #9525
- update mcoreddp call by @akoumpa :: PR: #9345
- mcore distOpt restore fix by @akoumpa :: PR: #9421
- vLLM Export Support by @apanteleev :: PR: #9381
- PL: Delete precision if using plugin. TODO switch to MegatronTrainerB… by @akoumpa :: PR: #9535
- extend get_gpt_layer_modelopt_spec to support MoE by @akoumpa :: PR: #9532
- fix mock data generation for legacy dataset by @dimapihtar :: PR: #9530
- add reset learning rate functionality by @dimapihtar :: PR: #9372
- Use closed-formula to round by multiple by @akoumpa :: PR: #9307
- GPU unit tests: Mark flaky tests to be fixed by @pablo-garay :: PR: #9559
- Consolidate gpt continue training script into pretraining script by @yaoyu-33 :: PR: #9413
- Enable encoder adapters for Canary and MultiTaskAED models by @titu1994 :: PR: #9409
- PTQ refinements by @janekl :: PR: #9574
- Add ModelOpt QAT example for Llama2 SFT model by @kevalmorabia97 :: PR: #9326
- Multimodal projection layer adapter fix for PP>1 by @paul-gibbons :: PR: #9445
- Add offline quantization script for QLoRA deployment by @cuichenx :: PR: #9455
- Make QLoRA more model-agnostic by @cuichenx :: PR: #9488
- Set n_gpu to None in nemo export by @oyilmaz-nvidia :: PR: #9593
- [NeMo-UX] Fix Megatron-optimizer by @marcromeyn :: PR: #9599
- Chat template support for megatron_gpt_eval.py by @akoumpa :: PR: #9354
- [NeMo-UX] Add PEFT by @cuichenx :: PR: #9490
- Alit/mamba tmp by @JRD971000 :: PR: #9612
- Enable MCore checkpointing optimizations by @mikolajblaz :: PR: #9505
- Change mixtral moe key name for trt-llm by @oyilmaz-nvidia :: PR: #9620
- fix ckpt load bug by @dimapihtar :: PR: #9621
- Alit/mamba by @JRD971000 :: PR: #9575
- Unwrap ckpt_io for model opt (async save) by @mikolajblaz :: PR: #9622
- MCore T5 support for NeMo - Training by @huvunvidia :: PR: #9432
- [Nemo-UX] Expose transformer_layer_spec inside GPTConfig by @marcromeyn :: PR: #9592
- Update NeMo Clip to Use MCore Modules by @yaoyu-33 :: PR: #9594
- Mistral + Mixtral Support for NeVa by @paul-gibbons :: PR: #9459
- Adding support for mcore generate by @shanmugamr1992 :: PR: #9566
- Improve error messaging during trt-llm export by @oyilmaz-nvidia :: PR: #9638
- [Cherrypick] support lora when kv_channel != hidden_size / num_heads by @cuichenx :: PR: #9644
- Parametrize FPS group by @mikolajblaz :: PR: #9648
- Cherry-pick megatron export fix from main by @borisfom :: PR: #9643
- add documentation for reset_lr feature by @dimapihta
- chore: Pin branch in notebooks by @ko3n1g :: PR: #9697
- Cherry pick: LITA Integration by @Slyne :: PR: #9684
- SDXL improvements (and support for Draft+) by @rohitrango :: PR: #9654
- Gemma 2 by @cuichenx :: PR: #9672
- Allows non-strict load with distributed checkpoints by @mikolajblaz :: PR: #9613
- refactor: notebook branch release by @ko3n1g :: PR: #9711
- [NeMo-UX] Make TE and Apex dependencies optional by @ashors1 :: PR: #9550
- Alit/r2.0.0 by @JRD971000 :: PR: #9718
- Manually cherry-pick from PR 9679 (PR to main - Support SFT/Eval/PEFT for mcore T5) by @huvunvidia :: PR: #9737
- In framework export by @oyilmaz-nvidia :: PR: #9658
- T5 changes based on mcore changes by @pablo-garay :: PR: #9829
- [NeMo-UX] Use single instance of loss reductions in GPTModel by @hemildesai :: PR: #9801
- deprecate NeMo NLP tutorial by @dimapihtar :: PR: #9864
- Disable nvFuser setup with PyTorch 23.11 and later by @athitten :: PR: #9837
- make torch_dist ckpt strategy as default by @dimapihtar :: PR: #9852
- add rampup bs documentation by @dimapihtar :: PR: #9884
- copy of #9576 by @dimapihtar :: PR: #9986
- Support Nvidia Torch and Arch versions by @thomasdhc :: PR: #9897
- Bug fix for pooler causing dist checkpointing exception by @shanmugamr1992 :: PR: #10008

</details>

#### Export

<details><summary>Changelog</summary>

- Update nemo.export module for quantized models by @janekl :: PR: #9218
- Add save option to the TRT-LLM export test script by @oyilmaz-nvidia :: PR: #9221
- Add TRT-LLM params like max_num_tokens and opt_num_tokens by @oyilmaz-nvidia :: PR: #9210
- TRT-LLM Export Code Cleanup by @oyilmaz-nvidia :: PR: #9270
- Re-org export code by @oyilmaz-nvidia :: PR: #9353
- Use TensorRT-LLM native parameter names in nemo.export module by @janekl :: PR: #9424
- TRT-LLM 0.10 Update by @oyilmaz-nvidia :: PR: #9402
- vLLM Export Support by @apanteleev :: PR: #9381
- Add page context fmha option in TensorRTLLM export by @meatybobby :: PR: #9526
- Test C++ runtime on demand in nemo_export.py to avoid possible OOMs by @janekl :: PR: #9544
- Fix nemo export test by @oyilmaz-nvidia :: PR: #9547
- Add tps and pps params to the export script by @oyilmaz-nvidia :: PR: #9558
- Add Multimodal Exporter by @meatybobby :: PR: #9256
- Set n_gpu to None in nemo export by @oyilmaz-nvidia :: PR: #9593
- Inflight nemo model export support by @JimmyZhang12 :: PR: #9527
- vLLM Export Improvements by @apanteleev :: PR: #9596
- Akoumparouli/nemo ux mixtral export by @akoumpa :: PR: #9603
- Change mixtral moe key name for trt-llm by @oyilmaz-nvidia :: PR: #9620
- Fix the arguments  of forward_for_export function in msdd_models by @tango4j :: PR: #9624
- Improve error messaging during trt-llm export by @oyilmaz-nvidia :: PR: #9638
- Cherry-pick megatron export fix from main by @borisfom :: PR: #9643
- In framework export by @oyilmaz-nvidia :: PR: #9658
- Add missing imports for torch dist ckpt in export by @oyilmaz-nvidia :: PR: #9826~

</details>

#### Bugfixes

<details><summary>Changelog</summary>

- use get with fallback when reading checkpoint_callback_params by @akoumpa :: PR: #9223
- fix import by @akoumpa :: PR: #9240
- Remove .nemo instead of renaming by @mikolajblaz :: PR: #9281
- call set_expert_model_parallel_world_size instead of set_cpu_expert_m… by @akoumpa :: PR: #9275
- Fix typos in Mixtral NeMo->HF and Starcoder2 NeMo->HF conversion scripts by @evellasques :: PR: #9325
- Skip sequence_parallel allreduce when using Mcore DistOpt by @akoumpa :: PR: #9344
- Add OpenAI format response to r2.0.0rc1 by @athitten :: PR: #9796
- [NeMo UX] Support generating datasets using different train/valid/test distributions by @ashors1 :: PR: #9771
- Add missing imports for torch dist ckpt in export by @oyilmaz-nvidia :: PR: #9826

</details>

#### General Improvements

<details><summary>Changelog</summary>

- [Nemo CICD] run_cicd_for_release_branches_also by @pablo-garay :: PR: #9213
- rename paths2audiofiles to audio by @github-actions[bot] :: PR: #9220
- Fix ASR_Context_Biasing.ipynb contains FileNotFoundError by @github-actions[bot] :: PR: #9234
- ci: Remove duplicated job by @ko3n1g :: PR: #9258
- Fix document links by @yaoyu-33 :: PR: #9260
- Pin transformers by @github-actions[bot] :: PR: #9273
- Fix loading github raw images on notebook by @github-actions[bot] :: PR: #9283
- Accept None as an argument to decoder_lengths in GreedyBatchedCTCInfer::forward by @github-actions[bot] :: PR: #9278
- Refactor Sequence Packing Script by @cuichenx :: PR: #9271
- [Nemo-UX] Move code to collections + fix some small bugs by @marcromeyn :: PR: #9277
- Fix typo in HF tutorial by @github-actions[bot] :: PR: #9304
- Expand documentation for data parallelism and distributed optimizer by @timmoon10 :: PR: #9227
- Install alerting by @ko3n1g :: PR: #9311
- typos by @github-actions[bot] :: PR: #9315
- FP8 feature documentation by @ksivaman :: PR: #9265
- [Nemo CICD] Comment out flaky tests by @pablo-garay :: PR: #9333
- Fixed typos in README.rst by @gdevakumar :: PR: #9322
- Update README.rst to clarify installation via Conda by @SimonCW :: PR: #9323
- [Nemo CICD] update flaky test by @pablo-garay :: PR: #9339
- fix lora and ptuning and isort/black by @github-actions[bot] :: PR: #9295
- Fix P-tuning for Llama based models by @github-actions[bot] :: PR: #9300
- add large model stable training fix and contrastive loss update for variable seq by @github-actions[bot] :: PR: #9348
- Guard cuda memory allocator update by @github-actions[bot] :: PR: #9313
- [Nemo CICD] Remove unnecessary commented out code by @pablo-garay :: PR: #9364
- Update Gemma conversion script by @yaoyu-33 :: PR: #9365
- Fix GreedyBatchedCTCInfer regression from GreedyCTCInfer. (#9347) by @github-actions[bot] :: PR: #9371
- Re-enable cuda graphs in training modes. by @github-actions[bot] :: PR: #9343
- fix typo infer_seq_lenght -> infer_seq_length by @akoumpa :: PR: #9370
- Make a backward compatibility for old MSDD configs in label models by @github-actions[bot] :: PR: #9378
- Dgalvez/fix greedy batch strategy name r2.0.0rc0 by @github-actions[bot] :: PR: #9253
- Update README.rst by @jgerh :: PR: #9393
- Force diarizer to use CUDA if cuda is available and if device=None. by @github-actions[bot] :: PR: #9390
- ci: Properly catch failed tests by introduction of workflow templates by @ko3n1g :: PR: #9324
- Fix T5 G2P Input and Output Types by @github-actions[bot] :: PR: #9269
- Huvu/rag pipeline citest by @huvunvidia :: PR: #9384
- Fix circular import for MM dataprep notebook by @github-actions[bot] :: PR: #9292
- add check if num layers is divisible by pp size by @github-actions[bot] :: PR: #9298
- [Nemo CICD] timeouts fix by @pablo-garay :: PR: #9407
- [NeMo-UX] Removing un-used ModelConfig class by @marcromeyn :: PR: #9389
- Add tutorial for Llama-3-8B lora training and deployment by @shashank3959 :: PR: #9359
- [NeMo-UX] Removing default_path from ModelConnector by @marcromeyn :: PR: #9401
- Fix README by @ericharper :: PR: #9415
- [SD] Fix SD CUDA Graph Failure by @alpha0422 :: PR: #9319
- [NeMo-UX] Adding file-lock to Connector by @marcromeyn :: PR: #9400
- Add Dev Container Bug Report by @pablo-garay :: PR: #9430
- Akoumparouli/profiling docs by @akoumpa :: PR: #9420
- ci: Enrich notifications by @ko3n1g :: PR: #9412
- Fix failing RIR unit test with lhotse 1.24+ by @pzelasko :: PR: #9444
- [NeMo-UX] Adding support for mcore distributed optimizer by @marcromeyn :: PR: #9435
- Use ModelOpt build_tensorrt_llm for building engines for qnemo checkpoints by @janekl :: PR: #9452
- ci(notifications): Fix extraction of last 2K chars by @ko3n1g :: PR: #9450
- Update readme with mlperf news by @ericharper :: PR: #9457
- [NeMo-UX] Add nsys callback by @ashors1 :: PR: #9461
- [NeMo UX] Introducing optimizer module by @marcromeyn :: PR: #9454
- Fix minor import bug in deploy module by @oyilmaz-nvidia :: PR: #9463
- ci(notifications): Fetch all jobs by @ko3n1g :: PR: #9465
- Update build_dataset.py by @stevehuang52 :: PR: #9467
- bionemo: bn2/add pipelineparallel dtype by @skothenhill-nv :: PR: #9475
- [NeMo-UX] Integrate experiment manager features with NeMo-UX APIs by @ashors1 :: PR: #9460
- Add python_requires by @galv :: PR: #9431
- [NeMo-UX] Fixing imports of NeMoLogging, AutoResume & ModelCheckpoint by @marcromeyn :: PR: #9476
- Modelopt Refactor for SDXL Quantization by @suiyoubi :: PR: #9279
- [NeMo-UX] Fixing defaults in llm.train & Mistral7BModel by @marcromeyn :: PR: #9486
- In framework deploy using deploy script by @oyilmaz-nvidia :: PR: #9468
- [NeMo-UX] Integrate tokenizer import into model.import_ckpt by @marcromeyn :: PR: #9485
- append to file by @malay-nagda :: PR: #9483
- [NeMo-UX] Fix bug in import_ckpt by @marcromeyn :: PR: #9492
- Add nemotron news by @ericharper :: PR: #9510
- Add CICD test for Stable Diffusion by @michal2409 :: PR: #9464
- Akoumparouli/nemo ux mixtral by @akoumpa :: PR: #9446
- [NeMo-UX] Llama and Gemma by @cuichenx :: PR: #9528
- [NeMo-UX] minor logging bug fixes by @ashors1 :: PR: #9529
- Update neva conversion script from and to HF by @yaoyu-33 :: PR: #9296
- [Nemo-UX] IO fixes by @marcromeyn :: PR: #9512
- Fix lhotse tests for v1.24.2 by @pzelasko :: PR: #9546
- [Nemo CICD] Make GPU Unit Tests non-optional by @pablo-garay :: PR: #9551
- Add Python AIStore SDK to container and bump min Lhotse version by @pzelasko :: PR: #9537
- [NeMo-UX] Fix tokenizer IO by @marcromeyn :: PR: #9555
- [NeMo UX] Move mistral_7b.py to mistral.py by @akoumpa :: PR: #9545
- ci: Do not attempt to send slack on fork by @ko3n1g :: PR: #9556
- Fix SDXL incorrect name in Docs by @suiyoubi :: PR: #9534
- Bump PTL version by @athitten :: PR: #9557
- [Resiliency] Straggler detection by @jbieniusiewi :: PR: #9473
- [NeMo-UX] Switch to torch_dist as default distributed checkpointing backend by @ashors1 :: PR: #9541
- [NeMo-UX] Checkpointing bug fixes by @ashors1 :: PR: #9562
- Expose MCore path_to_cache option by @maanug-nv :: PR: #9570
- [NeMo-UX] Fix Trainer serialization by @marcromeyn :: PR: #9571
- Update click version requirement by @thomasdhc :: PR: #9580
- [Fault tolerance] Heartbeat detection by @maanug-nv :: PR: #9352
- [Nemo-UX] Add fabric-API for manual forward-pass by @marcromeyn :: PR: #9577
- [Nemo-UX] Add SDK-factories to llm-collection by @marcromeyn :: PR: #9589
- [NeMo-UX] Some improvements to NeMoLogger by @marcromeyn :: PR: #9591
- Set no_sync_func & grad_sync_fucn by @akoumpa :: PR: #9601
- [NeMo-UX] Fix nemo logger when trainer has no loggers by @ashors1 :: PR: #9607
- Fix the dictionary  format returned by the `scheduler` method by @sararb :: PR: #9609
- [NeMo-UX] Dataloading enhancements and bug fixes by @ashors1 :: PR: #9595
- Fix serialization of AutoResume by @sararb :: PR: #9616
- Jsonl support by @adityavavre :: PR: #9611
- Akoumparouli/mistral import instruct chat template fix by @akoumpa :: PR: #9567
- Remove .cuda calls, use device isntead by @akoumpa :: PR: #9602
- fix converter defautl args by @akoumpa :: PR: #9565
- fix: remove non_blocking from PTL's .cuda call by @akoumpa :: PR: #9618
- NeVA Minor Fixes by @yaoyu-33 :: PR: #9608
- [NeMo-UX] fix pretrianing data sizes and weights by @cuichenx :: PR: #9627
- [NeMo-UX] async checkpointing support by @ashors1 :: PR: #9466
- Change default parallel_save to False by @mikolajblaz :: PR: #9632
- Add REST API to deploy module by @athitten :: PR: #9539
- ci: Timeout per step, not job by @ko3n1g :: PR: #9635
- [NeMo-UX] Fix when optimizers are setup for PEFT by @marcromeyn :: PR: #9619
- [NeMo-UX] Fix pipeline parallel bug by @ashors1 :: PR: #9637
- Fixing import error fior llama-index (RAG pipeline) by @pablo-garay :: PR: #9662
- llama CI fix by @rohitrango :: PR: #9663
- [NeMo-UX] Make 'load_directly_on_device' configurable by @ashors1 :: PR: #9657
- [Nemo-UX] Including all trainable-params in a PEFT-checkpoint by @marcromeyn :: PR: #9650
- [NeMo-UX] Fix imports so local configuration of runs works again by @marcromeyn :: PR: #9690
- Set TE flag in legacy -> mcore conversion script by @terrykong :: PR: #9722
- Update starthere docs text by @erastorgueva-nv :: PR: #9724
- TorchAudio installation workaround for incorrect `PYTORCH_VERSION` variable by @artbataev :: PR: #9736
- [NeMo-UX] Match nemo 1's default behavior for drop_last and pad_samples_to_global_batch_size by @ashors1 :: PR: #9707
- add a bit more for timeout (#9702) by @pablo-garay :: PR: #9754
- Fix missing parallelisms by @maanug-nv :: PR: #9725
- update branch by @nithinraok :: PR: #9764
- Fix data preprocessing script by @cuichenx :: PR: #9759
- vLLM 0.5.1 update by @apanteleev :: PR: #9779
- upper bound hf-hub by @akoumpa :: PR: #9805
- Fix few issues and docs for neva and clip in r2.0.0rc1 by @yaoyu-33 :: PR: #9681
- add dummy vision and text transformer config (assumed mcore to be false) by @rohitrango :: PR: #9699
- fix lita bugs by @Slyne :: PR: #9810
- [NeMo-UX] Log `val_loss` by @ashors1 :: PR: #9814
- [NeMo-UX] Fix some dataloading bugs by @ashors1 :: PR: #9807
- [NeMo-UX] Adding recipes by @marcromeyn :: PR: #9720
- [NeMo-UX] Set async_save from strategy rather than ModelCheckpoint by @ashors1 :: PR: #9800
- Fix hf hub for 0.24+ by @titu1994 :: PR: #9806
- [NeMo-UX] Fix a minor bug with async checkpointing by @ashors1 :: PR: #9856
- [NeMo-UX] make progress bar easier to parse by @ashors1 :: PR: #9877
- Docs: add "Nemo Fundamentals" page by @erastorgueva-nv :: PR: #9835
- Create __init__.py by @stevehuang52 :: PR: #9892
- [NeMo-UX] Fixes to make PreemptionCallback work by @hemildesai :: PR: #9830
- Fix Docker build. Make Dockerfile consistent with CI by @artbataev :: PR: #9784
- Multimodal data prep notebook fix by @cuichenx :: PR: #9910
- [NeMo-UX] Add distributed checkpointing unit tests by @ashors1 :: PR: #9794
- r2.0.0rc1 fix for dist checkpoint loading by @yaoyu-33 :: PR: #9854
- [NeMo-UX] Rename sdk references to NeMo Run by @hemildesai :: PR: #9872
- [NeMo-UX] Fix some serialization bugs by @ashors1 :: PR: #9868
- add mixtral neva tutorial (moe + token fusion + siglip) by @paul-gibbons :: PR: #9926
- [NeMo-UX] Add more NeMo Logger tests by @ashors1 :: PR: #9795
- Akoumparouli/mixtral fixes for r2.0.0rc1 by @akoumpa :: PR: #9911
- R2.0.0rc1 clip fix by @Slyne :: PR: #9871
- [NeMo-UX] Add missing docstrings and update some defaults by @ashors1 :: PR: #9895
- Add REST service requirements.txt by @oyilmaz-nvidia :: PR: #9923
- add bert latest fix by @JRD971000 :: PR: #9921
- remove empy reconfigure_limit_batches by @akoumpa :: PR: #9934
- fix mem by @terrykong :: PR: #9964
- Run a sample query for a quantized model conditionally by @janekl :: PR: #9965
- Add pydantic-settings  by @oyilmaz-nvidia :: PR: #9961
- Resiliency features update by @jbieniusiewi :: PR: #9714
- [NeMo-UX] Wrap task config save in a try/except by @ashors1 :: PR: #9956
- [NeMo-UX] Update default PTL logging `save_dir` by @ashors1 :: PR: #9954
- Fix lita tutorial by @Slyne :: PR: #9980
- Add deploy and REST API support to NeMo 2.0 by @athitten :: PR: #9834
- ci: Allow changelog manual (#10156) by @ko3n1g :: PR: #10157
- docs: Add changelog by @ko3n1g :: PR: #10155
- add manifest file by @ko3n1g :: PR: #10161

</details>

## NVIDIA Neural Modules 2.0.0rc0

### Highlights

#### LLM and MM

##### Models

- Megatron Core RETRO
  - Pre-training
  - Zero-shot Evaluation

- Pretraining, conversion, evaluation, SFT, and PEFT for:
  - Mixtral 8X22B
  - Llama 3
  - SpaceGemma

- Embedding Models Fine Tuning
  - Mistral
  - BERT

- BERT models
  - Context Parallel
  - Distributed checkpoint

- Video capabilities with NeVa

##### Performance

- Distributed Checkpointing
  - Torch native backend
  - Parallel read/write
  - Async write

- Multimodal LLM (LLAVA/NeVA)
  - Pipeline Parallelism support
  - Sequence packing support

##### Export

- Integration of Export & Deploy Modules into NeMo Framework container
  - Upgrade to TRT-LLM 0.9

#### Speech (ASR & TTS)

##### Models

- AED Multi Task Models (Canary) - Multi-Task Multi-Lingual Speech Recognition / Speech Translation model
- Multimodal Domain - Speech LLM supporting SALM Model
- Parakeet-tdt_ctc-1.1b Model - RTFx of > 1500 (can transcribe 1500 seconds of audio in 1 second)
- Audio Codec 16kHz Small - NeMo Neural Audio Codec for discretizing speech for use in LLMs
  - mel_codec_22khz_medium
  - mel_codec_44khz_medium

##### Perf Improvements

- Transcribe() upgrade - Enables one line transcribe with files, tensors, data loaders
- Frame looping algorithm for RNNT faster decoding - Improves Real Time Factor (RTF) by 2-3x
- Cuda Graphs + Label-Looping algorithm for RNN-T and TDT Decoding - Transducer Greedy decoding at over 1500x RTFx, on par with CTC Non-Autoregressive models
- Semi Sorted Batching support - External User contribution that speeds up training by 15-30%.

##### Customization

- Context biasing for CTC word stamping - Improve accuracy for custom vocabulary and pronunciation
  - Longform Inference
  - Longform inference support for AED models
- Transcription of multi-channel audio for AED models

##### Misc

- Upgraded webdataset - Speech and LLM / Multimodal unified container

### Detailed Changelogs

#### ASR

<details><summary>Changelog</summary>

- Enable using hybrid asr models in CTC Segmentation tool by @erastorgueva-nv :: PR: #8828
- TDT confidence fix by @GNroy :: PR: #8982
- Fix union type annotations for autodoc+mock-import rendering by @pzelasko :: PR: #8956
- NeMo dev doc restructure by @yaoyu-33 :: PR: #8896
- Improved random seed configuration for Lhotse dataloaders with docs by @pzelasko :: PR: #9001
- Fix #8948, allow preprocessor to be stream captured to a cuda graph when doing per_feature normalization by @galv :: PR: #8964
- [ASR] Support for transcription of multi-channel audio for AED models by @anteju :: PR: #9007
- Add ASR latest news by @titu1994 :: PR: #9073
- Fix docs errors and most warnings by @erastorgueva-nv :: PR: #9006
- PyTorch CUDA allocator optimization for dynamic batch shape dataloading in ASR by @pzelasko :: PR: #9061
- RNN-T and TDT inference: use CUDA graphs by default by @artbataev :: PR: #8972
- Fix #8891 by supported GPU-side batched CTC Greedy Decoding by @galv :: PR: #9100
- Update branch for notebooks and ci in release by @ericharper :: PR: #9189
- Enable CUDA graphs by default only for transcription by @artbataev :: PR: #9196
- rename paths2audiofiles to audio by @nithinraok :: PR: #9209
- Fix ASR_Context_Biasing.ipynb contains FileNotFoundError by @andrusenkoau :: PR: #9233
- Cherrypick: Support dataloader as input to `audio` for transcription (#9201) by @titu1994 :: PR: #9235
- Update Online_Offline_Microphone_VAD_Demo.ipynb by @stevehuang52 :: PR: #9252
- Dgalvez/fix greedy batch strategy name r2.0.0rc0 by @galv :: PR: #9243
- Accept None as an argument to decoder_lengths in GreedyBatchedCTCInfer::forward by @galv :: PR: #9246
- Fix loading github raw images on notebook by @nithinraok :: PR: #9282
- typos by @nithinraok :: PR: #9314
- Re-enable cuda graphs in training modes. by @galv :: PR: #9338
- add large model stable training fix and contrastive loss update for variable seq by @nithinraok :: PR: #9259
- Fix conv1d package in r2.0.0rc0  by @pablo-garay :: PR: #9369
- Fix GreedyBatchedCTCInfer regression from GreedyCTCInfer. (#9347) by @titu1994 :: PR: #9350
- Make a backward compatibility for old MSDD configs in label models by @tango4j :: PR: #9377
- Force diarizer to use CUDA if cuda is available and if device=None. by @tango4j :: PR: #9380

</details>

#### TTS

<details><summary>Changelog</summary>

- [TTS] Add tutorial for training audio codecs by @rlangman :: PR: #8723
- Update radtts.py by @blisc :: PR: #9097
- [Nemo CICD] RADTTS test optional by @pablo-garay :: PR: #9112
- Remove Radtts CI test by @blisc :: PR: #9144
- Fix T5 G2P Input and Output Types by @blisc :: PR: #9224

</details>

#### LLM and MM

<details><summary>Changelog</summary>

- Rachitg/dpa by @rachitgarg91 :: PR: #8911
- Remove precision args in trainer due to PTL update by @yaoyu-33 :: PR: #8908
- Huvu/mcore retro by @huvunvidia :: PR: #8861
- fsdp tp > 1 bug fix by @dimapihtar :: PR: #8947
- Fix memory leak at loss func by @minitu :: PR: #8868
- change the condition for get qkv tensor from linear_qkv output in mcoremixin by @HuiyingLi :: PR: #8965
- Add safety checks for 'data' key in MegatronGPTModel cfg by @HuiyingLi :: PR: #8991
- [NeMo-UX] Adding MegatronParallel by @cuichenx :: PR: #8987
- Skip top_p computations when set to 1.0 by @odelalleau :: PR: #8905
- Gemma bug by @cuichenx :: PR: #8962
- [NeMo-UX] Adding megatron strategy by @marcromeyn :: PR: #8995
- Quantized checkpoint support in export and deploy modules by @janekl :: PR: #8859
- add geglu to mlp swap by @JRD971000 :: PR: #8999
- add timeout for new_group by @acphile :: PR: #8998
- Zero-shot evaluation pipeline for mcore RETRO by @huvunvidia :: PR: #8941
- Added fusion for squared relu by @sanandaraj5597 :: PR: #8963
- Developer Documents for mcore RETRO by @huvunvidia :: PR: #9026
- [NeMo-UX] Adding GPTModel & MockDataModule by @marcromeyn :: PR: #9011
- Adding unit test for mcore RETRO model by @huvunvidia :: PR: #9022
- docs and simplification of cmd args by @arendu :: PR: #8979
- [NeMo-UX] Add checkpoint-io to MegatronStrategy by @marcromeyn :: PR: #9057
- Enable Sequence Packing and Pipeline Parallel in NeVA by @yaoyu-33 :: PR: #8957
- Mingyuanm/add back fp8 support to sd by @Victor49152 :: PR: #9070
- unfused lora by @arendu :: PR: #9004
- Handle case where num_query_groups is set to null for LoRA config setup by @vysarge :: PR: #9075
- Alit/griffin by @JRD971000 :: PR: #9021
- Implement DistributedCheckpointIO by @mikolajblaz :: PR: #9016
- Video Neva Pretraining + Inference Implementation by @paul-gibbons :: PR: #9095
- HF to .nemo for Mixtral-8x22B-instruct by @akoumpa :: PR: #9060
- mcore ds updates by @dimapihtar :: PR: #8951
- Alit/griffin perf by @JRD971000 :: PR: #9107
- Add assert for max_steps to be positive in MegatronGPTSFTModel by @athitten :: PR: #9110
- Extend sequence length padding for GPT SFT to account for context parallel by @vysarge :: PR: #8869
- Update gpt dataset config parameter for mock by @thomasdhc :: PR: #9118
- Add Mcore DistributedDataParallel and distributed optimizer into Nemo by @gdengk :: PR: #9034
- Revert "Add assert for max_steps to be positive in MegatronGPTSFTMode… by @pablo-garay :: PR: #9128
- scripts to convert HF lora to nemo by @arendu :: PR: #9102
- Prevent duplicated checkpoints by @mikolajblaz :: PR: #9015
- add TN/ITN link in speech tools list by @erastorgueva-nv :: PR: #9142
- Cleanup deprecated files and temporary changes by @cuichenx :: PR: #9088
- Use DP+CP groups as the FSDP sharding domain by @erhoo82 :: PR: #9145
- CUDA memory profile by @erhoo82 :: PR: #9096
- Fix missing func for T5 model by @gdengk :: PR: #9141
- Add knob for load_directly_on_device by @mikolajblaz :: PR: #9125
- Revert rope fusion defaults by @cuichenx :: PR: #9238
- Update nemo.export module for quantized models by @janekl :: PR: #9250
- Fix circular import for MM dataprep notebook by @cuichenx :: PR: #9287
- neva media_type + text generation default fix by @paul-gibbons :: PR: #9257
- fix lora and ptuning and isort/black by @oyilmaz-nvidia :: PR: #9290
- add check if num layers is divisible by pp size by @dimapihtar :: PR: #9208
- Fix P-tuning for Llama based models by @apanteleev :: PR: #9297
- add deprecation warnings by @pablo-garay :: PR: #9266
- move pooler under post_process by @dimapihtar :: PR: #9328
- add deprecation note for nmt by @dimapihtar :: PR: #9342
- Fix incorrect checkpoint removal logic (#9192) by @mikolajblaz :: PR: #9204
- fix fp16 precision issue by @dimapihtar :: PR: #9376
- Fix module.training for Neva in FusedAttn backward which causes nan by @yaoyu-33 :: PR: #8877

</details>

#### Export

<details><summary>Changelog</summary>

- Updates for TRT-LLM 0.9 by @oyilmaz-nvidia :: PR: #8873
- Mingyuanm/sdxl export by @Victor49152 :: PR: #8926
- Avoid unpacking NeMo checkpoints before exporting to TRT-LLM by @apanteleev :: PR: #8866
- Update gemma for trt-llm 0.9 by @oyilmaz-nvidia :: PR: #8974
- TRT-LLM export P-tuning related fixes by @apanteleev :: PR: #8863

</details>

#### General Improvements

<details><summary>Changelog</summary>

- Update package info by @ericharper :: PR: #8793
- [Nemo CICD] Update mcore 4.13.24 by @pablo-garay :: PR: #8917
- Akoumparouli/low mem mixtral ckpt converter by @akoumpa :: PR: #8895
- Adding RETRO tests to Action Tests (cicd-main.yml)  by @huvunvidia :: PR: #8942
- Akoumparouli/fix sd train 2 by @akoumpa :: PR: #8883
- Update te install for jenkins by @ericharper :: PR: #8954
- [Nemo CICD] Add last job depending on others for blocking check by @pablo-garay :: PR: #8959
- Minor quantization pipeline updates by @janekl :: PR: #8924
- Fix External CLIP Converter by @yaoyu-33 :: PR: #8960
- PP support in LoRA merge script by @cuichenx :: PR: #8934
- Update PR template by @ericharper :: PR: #8978
- Update Latest News by @shashank3959 :: PR: #8837
- Fix incorrect link to latest news in README by @shashank3959 :: PR: #8985
- Update dependency install for LLM and MM by @ericharper :: PR: #8990
- Temporarily remove mcore dep by @ericharper :: PR: #9010
- [Nemo CICD] further specialize runners for more parallelism by @pablo-garay :: PR: #9036
- Update mm dataprep notebook based on feedback by @cuichenx :: PR: #9029
- Fix import in lora merge script by @cuichenx :: PR: #9032
- [Nemo CICD] Run when labeled:Run CICD by @pablo-garay :: PR: #9044
- [Nemo CICD] Add tag/label for 1-gpu runner by @pablo-garay :: PR: #9046
- [Nemo CICD] checkout v4 by @pablo-garay :: PR: #9048
- [Nemo CICD] Remove temp test change by @pablo-garay :: PR: #9049
- remove in-place addition for dreambooth train with text encoder by @Victor49152 :: PR: #8825
- Mingyuanm/sdxl quantization notebook by @Victor49152 :: PR: #9042
- [Nemo CICD] Trigger on comment issued by @pablo-garay :: PR: #9062
- zarr ckpt to torch_dist ckpt converter by @dimapihtar :: PR: #8842
- Restore PTQ tests for Llama2 (reopened) by @janekl :: PR: #9064
- add clip H config by @JRD971000 :: PR: #9082
- [NeMo-UX] Add mixed-precision plugin by @marcromeyn :: PR: #9065
- Comment baichuan test and update pr template by @ericharper :: PR: #9085
- Add safe extraction of nemo tar files by @athitten :: PR: #8976
- Improved `shard_id` parsing in `LazyNemoTarredIterator`, enables AIS dataloading by @pzelasko :: PR: #9077
- [NeMo-UX] Add mistral-7b model by @marcromeyn :: PR: #9066
- Llama3 Conversion Script Update by @suiyoubi :: PR: #9089
- dehardcode test string by @JimmyZhang12 :: PR: #8865
- [Nemo CICD] Try trigger cicd run on comment by @pablo-garay :: PR: #9111
- Lhotse dataloading: RIR augmentation and nemo/tarred input support for RIR and noise aug by @pzelasko :: PR: #9109
- mixtral evaluation PR by @Slyne :: PR: #8989
- [Nemo CICD] Revert: run GHA cicd on comment by @pablo-garay :: PR: #9119
- [Nemo CICD] Comment out flaky test: running too long by @pablo-garay :: PR: #9123
- [Nemo CICD] Add timeout to unit tests by @pablo-garay :: PR: #9132
- [Nemo CICD] Indicate optional test in name (prefix) by @pablo-garay :: PR: #9139
- video neva null image+video folder path fix by @paul-gibbons :: PR: #9116
- [NeMo-UX] Add data module by @cuichenx :: PR: #9133
- NeMo Inference Requirements by @oyilmaz-nvidia :: PR: #9093
- Remove debug print by @maanug-nv :: PR: #9074
- Remove legacy CI by @pablo-garay :: PR: #9149
- Update support for push_to_hf_hub() by @titu1994 :: PR: #9159
- [Nemo CICD] comment out flaky PTQ tests by @pablo-garay :: PR: #9160
- Update branch by @ericharper :: PR: #9211
- dist adam transpose fix by @dimapihtar :: PR: #9239
- [Nemo CICD] Increase time limit for Speech_Checkpoints_tests (#9186) by @pablo-garay :: PR: #9247
- Pin transformers by @ericharper :: PR: #9261
- Fix typo in HF tutorial by @titu1994 :: PR: #9302

</details>

## NVIDIA Neural Modules 1.23.0

### Highlights

#### Models

##### Nvidia Starcoder 2 - 15B

- Announcement - <https://developer.nvidia.com/blog/unlock-your-llm-coding-potential-with-starcoder2/>
- AI Foundation Model Inference  - <https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-foundation/models/starcoder2-15b>
- <https://huggingface.co/bigcode/starcoder2-15b>

##### NeMo Canary

Announcement - <https://nvidia.github.io/NeMo/blogs/2024/2024-02-canary/>

- <https://huggingface.co/nvidia/canary-1b>

#### NeMo LLM

- Falcon
- Code Llama
- StarCoder
- GPT perf improvements
- Context parallelism
- Mistral
- Mixtral (without expert parallelism)
- Mcore GPT Dataset integration

#### NeMo MM

- CLIP
- Stable Diffusion (supporting LoRA)
- Imagen
- ControlNet (for SD)
- Instruct pix2pix (for SD)
- LLAVA
- NeVA
- DreamFusion++
- NSFW filtering

#### NeMo ASR

- Lhotse Dataloading support #7880
- Canary: Multi task multi lingual ASR #8242
- LongForm Audio for Diarization #7737
- Faster algorithm for RNN-T Greedy #7926
- Cache-Aware streaming notebook #8296

#### NeMo TTS

#### NeMo Vision

#### Known Issues

##### ASR

###### RNNT WER calculation when fused batch size > 1 during validation / test step()

Previously, the RNNT metric was stateful while the CTC one was not ([r1.22.0](https://github.com/NVIDIA/NeMo/blob/r1.22.0/nemo/collections/asr/metrics/rnnt_wer_bpe.py#L419-L420), [r1.23.0](https://github.com/NVIDIA/NeMo/blob/r1.23.0/nemo/collections/asr/metrics/wer.py#L333))

Therefore this calculation in the RNNT joint for fused operation worked properly. However with the unification of metrics in r1.23.0, a bug was introduced where only the last sub-batch of metrics calculates the scores and does not accumulate. This is patched via <https://github.com/NVIDIA/NeMo/pull/8587> and will be fixed in the next release.

__Workaround__: Explicitly disable fused batch size during inference using the following command

```python
from omegaconf import open_dict
model = ...
decoding_cfg = model.cfg.decoding
with open_dict(decoding_cfg):
  decoding_cfg.fused_batch_size = -1
model.change_decoding_strategy(decoding_cfg)
```

Note: This bug does not affect scores calculated via model.transcribe() (since it does not calculate metrics during inference, just text), or using the `transcribe_speech.py` or `speech_to_text_eval.py` in `examples/asr`.

###### Two failing unit tests due to a change in expected results, caused by lhotse version update

#### Container

For additional information regarding NeMo containers, please visit: <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo>

`docker pull nvcr.io/nvidia/nemo:24.01.speech`

#### ASR

<details><summary>Changelog</summary>

- Update link to yaml file in ASR_with_Transducers.ipynb by @Faith-Nchifor :: PR: #8014
- Use convert_hf_dataset_to_nemo by @karpnv :: PR: #8017
- Update asr_language_modeling.rst: Add a missing word by @martin0258 :: PR: #8007
- spelling mistake by @orena1 :: PR: #7903
- update asr eval by @stevehuang52 :: PR: #8045
- fix noise aug by @stevehuang52 :: PR: #8057
- Various fixes for typos and urls by @titu1994 :: PR: #8066
- [Fix] Increase length check tolerance to prevent test failing by @anteju :: PR: #8067
- Add text metrics to asr eval by @stevehuang52 :: PR: #8087
- fix device setting to allow using accelerator cpu by @orena1 :: PR: #8084
- .ctm in data simulator annotator compliant with RT-09 specification by @popcornell :: PR: #8004
- Fix AST eval by @stevehuang52 :: PR: #8112
- fix: numba.*_num_threads resets torch num_threads #8141 by @itzsimpl :: PR: #8145
- Update dependencies by @titu1994 :: PR: #8156
- NeMo + Lhotse integration by @pzelasko :: PR: #7880
- Speedup RNN-T greedy decoding by @artbataev :: PR: #7926
- [docker] Install k2 before NeMo for faster image rebuilding by @pzelasko :: PR: #8204
- [docs] Add --force_codec to tarred dataset creation examples by @pzelasko :: PR: #8227
- Temporarily use the previous RNN-T decoding algorithm as default by @artbataev :: PR: #8226
- Make TDT inference not require duration params by @hainan-xv :: PR: #8207
- Cache Aware Streaming tutorial notebook by @erastorgueva-nv :: PR: #8296
- fix path location and branch by @nithinraok :: PR: #8304
- Attention encoder-decoder models for multiple speech-to-text tasks  … by @titu1994 :: PR: #8324
- Remove asr webapp by @titu1994 :: PR: #8347
- remove _target_ at model level in aed model config [ASR] by @krishnacpuvvada :: PR: #8351
- Add change_vocabulary and save_tokenizers() support to Multitask ASR models by @titu1994 :: PR: #8357
- Change default beam size by @titu1994 :: PR: #8371
- adding jenkins test for speech_to_text_aed model by @krishnacpuvvada :: PR: #8368
- Add Finetuning tutorial with HF Datasets by @nithinraok :: PR: #8356
- wer fix by @tbartley94 :: PR: #8404
- add ensemble decoding fix by @nithinraok :: PR: #8427
- Update k2 by @artbataev :: PR: #8492

</details>

#### TTS

<details><summary>Changelog</summary>

- [TTS] Scale sampler steps by number of devices by @rlangman :: PR: #7947
- Add All Multimodal Source Code Part 2: Text to image, x to nerf by @yaoyu-33 :: PR: #7970
- [TTS] Add period discriminator and feature matching loss to codec recipe by @rlangman :: PR: #7884
- Added VectorQuantizer base class by @anteju :: PR: #8011

</details>

#### LLMS

<details><summary>Changelog</summary>

- Add interface to set NCCL options of each process group by @erhoo82 :: PR: #7923
- Support O2 training of PEFT and SFT by @cuichenx :: PR: #7971
- [NLP] Access scaler only in FP16 case by @janekl :: PR: #7916
- [NLP] Minor improvements in Llama conversion script by @janekl :: PR: #7978
- [NLP] Use helpers from utils_funcs.py in Llama conversion by @janekl :: PR: #7979
- [NLP] Remove replace_sampler_ddp (deprecated in Trainer) by @janekl :: PR: #7981
- Reworked MegatronPretrainingRandomBatchSampler to correctly handle epochs > 1 by @trias702 :: PR: #7920
- Remove deprecated arguments from TE's TransformerLayer by @jbaczek :: PR: #7917
- Add All Multimodal Source Code by @yaoyu-33 :: PR: #7791
- First draft of mcore bert model in NeMo by @shanmugamr1992 :: PR: #7814
- Support Falcon Variants (7B/40B/180B) in Mcore NeMo by @xuanzic :: PR: #7666
- FSDP + Tensor Parallelism by @erhoo82 :: PR: #7897
- Packed Sequence by @cuichenx :: PR: #7945
- Adding method back that was removed accidentally by @ericharper :: PR: #8038
- [NLP] ArtifactItem with init=True to make it debuggable by @janekl :: PR: #7980
- SFT patch: (1) enable sequence parallelism and (2) enable profile by @erhoo82 :: PR: #7963
- migration to PTL 2.0 for spellmapper model by @bene-ges :: PR: #7924
- Change the megatron config lr scheduler default and fix to change partitions script by @shan18 :: PR: #8094
- (1) Add SHARP interface to M-CORE, (2) use send/recv to send train loss to the first rank instead of b-cast by @erhoo82 :: PR: #7793
- Reconfigure limit_val_batches only for int by @athitten :: PR: #8099
- Fixing wrapper and moving it to base class by @shanmugamr1992 :: PR: #8055
- fix gated_linear_unit bug by @Agoniii :: PR: #8042
- Fix Adapter for MCore models by @cuichenx :: PR: #8124
- add war fix for sync issues by @gshennvm :: PR: #8130
- Improve PEFT UX by @cuichenx :: PR: #8131
- Enhance flexibility by passing callbacks as method argument by @michal2409 :: PR: #8015
- context parallelism by @xrennvidia :: PR: #7739
- Make pipelined TP comm overlap available with mcore by @erhoo82 :: PR: #8005
- remove deprecated scripts by @arendu :: PR: #8138
- adding OnlineSampleMapping by @arendu :: PR: #8137
- Add distopt support for FP8 params and BF16 optimizer state by @timmoon10 :: PR: #7909
- Revert adding OnlineSampleMapping by @pablo-garay :: PR: #8164
- Token count and sequence length logging for MegatronGPTSFTModel by @vysarge :: PR: #8136
- Use latest apex internal API by @jbaczek :: PR: #8129
- tune specific params in the base model by @arendu :: PR: #7745
- Virtual pipeline parallel support for MegatronGPTSFTModel by @vysarge :: PR: #7964
- removed deprecated peft model by @arendu :: PR: #8183
- remove more deprecated files by @arendu :: PR: #8169
- Pre-generate cu_seqlens argmin and max_seqlen to remove host-to-device sync by @erhoo82 :: PR: #8108
- Add the interface to use SHARP to FSDP strategy by @erhoo82 :: PR: #8202
- Multimodal required NLP base model changes by @yaoyu-33 :: PR: #8188
- [NLP] Improve and unify loading state_dict for community models by @janekl :: PR: #7977
- Rename Finetuning Scripts by @cuichenx :: PR: #8201
- Final multimodal PR with our recent developments on MM side by @yaoyu-33 :: PR: #8127
- Add include_text parameter to SFT dataloaders by @Kipok :: PR: #8198
- Add random_seed argument to generate by @Kipok :: PR: #8162
- Added support for neptune logger by @harishankar-gopalan :: PR: #8210
- Pre-compute max_seqlen and cu_seqlens_argmin in all model-parallel cases by @erhoo82 :: PR: #8222
- Use PackedSeqParams in accordance with changes in Megatron-LM by @cuichenx :: PR: #8205
- Fix to peft & virtual pipeline parallel unsupported check by @vysarge :: PR: #8216
- Fixed the tp overlap switch by @sanandaraj5597 :: PR: #8195
- add knobs for rope/swiglu fusion by @lhb8125 :: PR: #8184
- Added sample cpu_offloading switch to YAML by @sanandaraj5597 :: PR: #8148
- Syncing random seed between ranks in generate by @Kipok :: PR: #8230
- add first_val_step to mcore scheduler by @JimmyZhang12 :: PR: #8150
- Correct padding for SFT input data to account for sequence parallel + TE's fp8 op dimension requirements by @vysarge :: PR: #8240
- Mistral 7b conversion script by @akoumpa :: PR: #8052
- switch to mcore dataset [with FIM support] by @dimapihtar :: PR: #8149
- Mixtral to NeMo conversion script. by @akoumpa :: PR: #8155
- fixes to accomendate mcore changes by @HuiyingLi :: PR: #8261
- Allow MegatronPretrainingRandomSampler to do multi-epoch training by @trias702 :: PR: #8239
- Add dist ckpt support for regular optimizers by @mikolajblaz :: PR: #7749
- add deallocate pipeline output optimization by @JimmyZhang12 :: PR: #8279
- Fix memory leak caused by context parallelism hanging references by omegaconf by @JimmyZhang12 :: PR: #8299
- distributed fused adam + rampup bs support by @dimapihtar :: PR: #8302
- Update PEFT Doc by @cuichenx :: PR: #8262
- Converter script fixes for mixtral/mistral by @akoumpa :: PR: #8272
- Keep max_seqlen and cu_seqlens_argmin for later micro-batches when PP>1 by @erhoo82 :: PR: #8334
- Enable megatron core loggers for GPT pretraining by @ashbhandare :: PR: #8354
- mcore ds fix by @dimapihtar :: PR: #8283
- release updates by @dimapihtar :: PR: #8378
- Mcore customization doc by @HuiyingLi :: PR: #8298
- updated link to pubmed by @nithinraok :: PR: #8402
- mcore customization doc minor fix by @HuiyingLi :: PR: #8421
- Fixing mcore bert for TP, PP and SP by @shanmugamr1992 :: PR: #8336
- Add settings to suppress bf16 compile errors in CI on V100 by @athitten :: PR: #8481
- MoE parameter passing by @akoumpa :: PR: #8255
- Add fp8 support for SD/Update notebook paths by @Victor49152 :: PR: #8489

</details>

#### NeMo Tools

<details><summary>Changelog</summary>

- SDE bugfix log by @Jorjeous :: PR: #8430

</details>

#### General Improvements

<details><summary>Changelog</summary>

- Add news section to README by @ericharper :: PR: #7984
- Fixing conversion script to work for code llama by @shanmugamr1992 :: PR: #7997
- Fix crash when converting to mcore a model using rotary embeddings by @odelalleau :: PR: #7998
- Added a procedure for Windows users, README by @Jorjeous :: PR: #7942
- Update manifest.py to speedup loading tarred datasets by @stevehuang52 :: PR: #7900
- [Fix] Fixed name of a test by @anteju :: PR: #7986
- Fix lora merge script by @cuichenx :: PR: #8113
- Support transcoding audio formats when saving tarred datasets (FLAC, OPUS) by @pzelasko :: PR: #8102
- README edit to change Apple Silicon install instructions (to fix a break introduced by pytorch 2) by @stephenmcconnachie :: PR: #8122
- Fixes NVIDIA/apex installation to not erroneously install the  pkg by @terrykong :: PR: #8126
- Graphviz fix by @GNroy :: PR: #7843
- Update README.rst by @fayejf :: PR: #8154
- Fix TP>1 issue for conversion script by @cuichenx :: PR: #8144
- Support torch jit script by @artbataev :: PR: #8027
- NeMo Multimodal Docs and Tests Initial PR by @yaoyu-33 :: PR: #8028
- Remove left-over prints in NeMo+Lhotse code by @pzelasko :: PR: #8180
- Upgrade to DLFW PyTorch 23.12 by @ericharper :: PR: #8163
- Add Lhotse support for  key in NeMo manifests by @pzelasko :: PR: #8197
- Fix CPU Initialization and TP>1 for LoRA Merge Script by @cuichenx :: PR: #8199
- Add support in Neural Typecheck to disable semantic checks by @titu1994 :: PR: #8212
- Pin lhotse=1.19.2 in r1.23.0 by @pzelasko :: PR: #8303
- Multimodal r1.23.0 bug fix  by @yaoyu-33 :: PR: #8315
- MCore dataset compatibility for tokenizers by @vysarge :: PR: #8390
- Update NFA video download link by @erastorgueva-nv :: PR: #8406
- Update MM Dataprep Tutorial by @cuichenx :: PR: #8410
- Fix dreambooth data sampler issue by @yaoyu-33 :: PR: #8400
- Fix a bug in CTM line processing function for multi-speaker data simulations by @tango4j :: PR: #8416
- Akoumparouli/mistral bugfix by @akoumpa :: PR: #8353
- pin to 0.5.0 by @ericharper :: PR: #8465
- Update NeMo Multimodal Requirements by @yaoyu-33 :: PR: #8515
- Fix link in multimodal dataprep tutorial by @cuichenx :: PR: #8517

</details>
