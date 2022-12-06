@Library('blossom-github-lib@master') 
import ipp.blossom.*

podTemplate(cloud:'sc-ipp-blossom-prod', yaml : """
apiVersion: v1
kind: Pod
metadata:
  labels:
    some-label: some-label-value
spec:
  volumes:
  - name: scratch
    nfs:
      server: ipp1-cdot01-col01
      path: /vol/scratch1/scratch.okuchaiev_blossom
  containers:
  - name: latestdlfw
    image: nvcr.io/nvidia/pytorch:22.11-py3
    command:
    - cat
    volumeMounts:
    - name: scratch
      mountPath: /testdata
    resources:
          limits:
             nvidia.com/gpu: 2
    restartPolicy: Never
    backoffLimit: 4
    tty: true
    shm-size: 32g
  nodeSelector:
    kubernetes.io/os: linux
    nvidia.com/gpu_type: "Tesla_T4x4"
    nvidia.com/node_type: gpu_tester
    nvidia.com/driver_version: "510.20"
"""
)   {
      node(POD_LABEL) {
          def githubHelper
          stage('Get Token') {
              withCredentials([usernamePassword(credentialsId: 'GHAtoken', passwordVariable: 'GIT_PASSWORD', usernameVariable: 'GIT_USERNAME')]) {
                  // create new instance of helper object
                  githubHelper = GithubHelper.getInstance("${GIT_PASSWORD}", githubData)
              }
              
          }
          def stageName = '' 
          try {
              currentBuild.description = githubHelper.getBuildDescription()
              container('latestdlfw') {
                stage('Code checkout') {
                    // update status on github
                    githubHelper.updateCommitStatus("$BUILD_URL", "$stageName Running", GitHubCommitState.PENDING)
                    checkout changelog: true, poll: true, scm: [$class: 'GitSCM', branches: [[name: "pr/"+githubHelper.getPRNumber()]],
                    doGenerateSubmoduleConfigurations: false,
                    submoduleCfg: [],
                    userRemoteConfigs: [[credentialsId: 'github-token', url: githubHelper.getCloneUrl(), refspec: '+refs/pull/*/head:refs/remotes/origin/pr/*']]]              
                }

                stage('Code Style') {
                        sh "apt-get update && \
                            apt-get install -y bc && \
                            nvidia-smi && \
                            pip install -r requirements/requirements_test.txt && \
                            python setup.py style && ls -l /testdata/TestData && ln -s /testdata/TestData /home/TestData && \
                            ls -l /home && ls -l /home/TestData"
                }
                
                stage('Installation') {
                  sh "git config --global --add safe.directory '*' && nvidia-smi && ./reinstall.sh release"
                }

                stage('L0: GPU unit tests') {
                            sh "NEMO_NUMBA_MINVER=0.53 pytest -m 'not pleasefixme'"
                }

                parallel( //USE CUDA_VISIBLE_DEVICES to execute 2 single GPU tests in parallel here
                [
                    "L1: NMT Training Pre-LN": { sh 'CUDA_VISIBLE_DEVICES=0 python examples/nlp/machine_translation/enc_dec_nmt.py \
                            --config-path=conf \
                            --config-name=aayn_base \
                            do_testing=true \
                            model.train_ds.src_file_name=/testdata/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
                            model.train_ds.tgt_file_name=/testdata/TestData/nlp/nmt/toy_data/wmt14-de-en.ref \
                            model.validation_ds.src_file_name=/testdata/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
                            model.validation_ds.tgt_file_name=/testdata/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
                            model.test_ds.src_file_name=/testdata/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
                            model.test_ds.tgt_file_name=/testdata/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
                            model.encoder_tokenizer.tokenizer_model=/testdata/TestData/nlp/nmt/toy_data/tt_tokenizer.BPE.4096.model \
                            model.decoder_tokenizer.tokenizer_model=/testdata/TestData/nlp/nmt/toy_data/tt_tokenizer.BPE.4096.model \
                            model.encoder.pre_ln=true \
                            model.decoder.pre_ln=true \
                            trainer.devices=[0] \
                            trainer.accelerator="gpu" \
                            +trainer.fast_dev_run=true \
                            +trainer.limit_test_batches=2 \
                            exp_manager=null \
                            '},
                    "L1: Speech to text": { sh 'CUDA_VISIBLE_DEVICES=1 python examples/asr/asr_ctc/speech_to_text_ctc.py \
                            model.train_ds.manifest_filepath=/testdata/TestData/an4_dataset/an4_train.json \
                            model.validation_ds.manifest_filepath=/testdata/TestData/an4_dataset/an4_val.json \
                            trainer.devices=[0] \
                            trainer.accelerator="gpu" \
                            +trainer.fast_dev_run=True \
                            exp_manager=null \
                            '}
                ]
                )//end of parallel
              }
              githubHelper.updateCommitStatus("$BUILD_URL", "Complete", GitHubCommitState.SUCCESS)
          }
          catch (Exception ex){
              currentBuild.result = 'FAILURE'
              println ex
              githubHelper.updateCommitStatus("$BUILD_URL", "$stageName Failed", GitHubCommitState.FAILURE)
          }
          
      }
  }