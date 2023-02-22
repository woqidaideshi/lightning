#!/bin/bash
# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
set -e
# THIS FILE ASSUMES IT IS RUN INSIDE THE tests/tests_pytorch DIRECTORY

# this environment variable allows special tests to run
export PL_RUN_STANDALONE_TESTS=1

can_run_nvprof=$(python -c "import torch; print(torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8)")
if [[ $can_run_nvprof == "True" ]]; then
    echo "Running profilers/test_profiler.py::test_pytorch_profiler_nested_emit_nvtx"
    nvprof --profile-from-start off -o trace_name.prof -- python -m coverage run --source pytorch_lightning --append -m pytest --no-header profilers/test_profiler.py::test_pytorch_profiler_nested_emit_nvtx
fi

# test that a user can manually launch individual processes
echo "Running manual bagua launch test"
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
args="fit --trainer.accelerator gpu --trainer.devices 2 --trainer.strategy bagua --trainer.max_epochs=1 --trainer.limit_train_batches=1 --trainer.limit_val_batches=1 --trainer.limit_test_batches=1"
MASTER_ADDR="localhost" MASTER_PORT=1234 LOCAL_RANK=1 python strategies/scripts/cli_script.py ${args} &
MASTER_ADDR="localhost" MASTER_PORT=1234 LOCAL_RANK=0 python strategies/scripts/cli_script.py ${args}

# test that bagua can launched as a module (-m option)
echo "Running bagua example as module"
echo -----args: $args
python -m strategies.scripts.cli_script ${args}
