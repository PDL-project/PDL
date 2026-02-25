
for i in range(25):
    action_queue.append({'action':'Done'})
    action_queue.append({'action':'Done'})
    action_queue.append({'action':'Done'})
    time.sleep(0.1)

task_over = True
time.sleep(5)

# MAP-THOR evaluation metrics (replaces GCR-based evaluation)
exec_rate = float(success_exec) / float(total_exec) if total_exec > 0 else 0.0
if "agent_success_counts" in globals() and len(agent_success_counts) > 0:
    _max_cnt = max(agent_success_counts)
    _min_cnt = min(agent_success_counts)
    balance = (float(_min_cnt) / float(_max_cnt)) if _max_cnt > 0 else 0.0
else:
    balance = 0.0

if checker is not None:
    coverage       = checker.get_coverage()
    transport_rate = checker.get_transport_rate()
    finished       = bool(checker.check_success())
else:
    # checker unavailable (task_folder not mapped): fall back to execution rate only
    coverage       = 0.0
    transport_rate = 0.0
    finished       = False

print(
    f"Coverage:{coverage:.3f}, Transport Rate:{transport_rate:.3f}, "
    f"Finished:{finished}, Balance:{balance:.3f}, Exec:{exec_rate:.3f}"
)

generate_video()
