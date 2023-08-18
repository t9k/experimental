#!/bin/bash

invokeai-configure --yes --default_only --root_dir /workspace/invokeai
invokeai --web --host 0.0.0.0 --root_dir /workspace/invokeai
