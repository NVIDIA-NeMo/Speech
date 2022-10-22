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
    image: nvcr.io/nvidia/pytorch:22.09-py3
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
  nodeSelector:
    kubernetes.io/os: linux
    nvidia.com/gpu_type: "Tesla_T4x4"
    nvidia.com/node_type: gpu_tester
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

                stage('Parallel Stage') {
                parallel(
                [
                    "foo": stage('A') { sh "nvidia-smi" },
                    "bar": stage('B') { sh "ls -l" }
                ]
                )}

                stage('Code Style') {
                        sh "apt-get update && \
                            apt-get install -y bc && \
                            pip install -r requirements/requirements_test.txt && \
                            python setup.py style && ls -l /testdata/TestData"
                }
                
                stage('Installation') {
                  sh "git config --global --add safe.directory '*' && nvidia-smi && ./reinstall.sh release"
                }

                stage('L0: GPU unit tests') {
                            sh "NEMO_NUMBA_MINVER=0.53 pytest -m 'not pleasefixme and not torch_tts'"         
                }

                stage('L1: NMT Training Pre-LN') {
                            sh 'cd examples/nlp/machine_translation && \
                            CUDA_VISIBLE_DEVICES=0 python enc_dec_nmt.py \
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
                            '        
                          }
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