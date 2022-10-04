#!/usr/bin/env sh

config=$1
others=${@:2}

if ! command -v yq &> /dev/null; then
	echo "The command <yq> does not exist"
	exit
fi

nproc=$(yq e '.grid.proc_layout[0] * .grid.proc_layout[1]' $config)
mpirun -np $nproc shallow_water_demo $others $config

