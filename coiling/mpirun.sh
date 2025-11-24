#!/bin/zsh

# Assumes script is run from project root
mpirun -np 8 ./cmake-build-release/hemoflow ./coiling/sim_config.xml