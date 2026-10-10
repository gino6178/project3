#!/bin/bash
# Workers, one per GPU, drain ~/ov/queue.txt through remote_object.sh; a new object appended to the queue
# is picked up by the next free GPU.  A worker waits up to 30 minutes on an empty queue before it exits (a batch is
# submitted one object at a time); local_object.sh restarts the scheduler if no worker is left.
echo $$ > ~/ov/sched_ai.pid
worker(){ g=$1; while true; do
  until mkdir ~/ov/.lock 2>/dev/null; do sleep 1; done
  o=$(head -n 1 ~/ov/queue.txt 2>/dev/null); [ -n "$o" ] && sed -i '1d' ~/ov/queue.txt; rmdir ~/ov/.lock
  if [ -z "$o" ]; then idle=$((${idle:-0}+1)); [ $idle -gt 60 ] && break; sleep 30; continue; fi; idle=0   # wait 30 min for late submissions
  echo "$(date +%H:%M) gpu$g $o" >> ~/ov/sched.log; bash ~/repro/frames/remote_object.sh $o $g; echo "$(date +%H:%M) gpu$g $o done" >> ~/ov/sched.log
done; }
for g in ${GPUS:-0 1 2 3}; do worker $g & done; wait; rm -f ~/ov/sched_ai.pid; echo "$(date +%H:%M) queue empty" >> ~/ov/sched.log
