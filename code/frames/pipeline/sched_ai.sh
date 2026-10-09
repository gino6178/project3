#!/bin/bash
# Workers, one per GPU, drain ~/ov/queue.txt through remote_object.sh; a new object appended to the queue
# is picked up by the next free GPU.  Exits when the queue is empty; local_object.sh restarts it.
echo $$ > ~/ov/sched_ai.pid
worker(){ g=$1; while true; do
  until mkdir ~/ov/.lock 2>/dev/null; do sleep 1; done
  o=$(head -n 1 ~/ov/queue.txt 2>/dev/null); [ -n "$o" ] && sed -i '1d' ~/ov/queue.txt; rmdir ~/ov/.lock
  [ -z "$o" ] && break
  echo "$(date +%H:%M) gpu$g $o" >> ~/ov/sched.log; bash ~/repro/frames/remote_object.sh $o $g; echo "$(date +%H:%M) gpu$g $o done" >> ~/ov/sched.log
done; }
for g in ${GPUS:-0 1 2 3}; do worker $g & done; wait; rm -f ~/ov/sched_ai.pid; echo "$(date +%H:%M) queue empty" >> ~/ov/sched.log
