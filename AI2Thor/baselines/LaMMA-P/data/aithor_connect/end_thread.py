
for i in range(25):
    action_queue.append({'action':'Done'})
    action_queue.append({'action':'Done'})
    action_queue.append({'action':'Done'})
    time.sleep(0.1)

task_over = True
time.sleep(5)

# MAP-THOR evaluation metrics (replaces GCR-based evaluation)
exec_rate = float(success_exec) / float(total_exec) if total_exec > 0 else 0.0

if checker is not None:
    coverage       = checker.get_coverage()
    transport_rate = checker.get_transport_rate()
    tc             = 1 if checker.check_success() else 0
else:
    # checker unavailable (task_folder not mapped): fall back to execution rate only
    coverage       = 0.0
    transport_rate = 0.0
    tc             = 0

sr = tc  # SR = 1 iff all subtasks completed

print(f"SR:{sr}, TC:{tc}, Coverage:{coverage:.3f}, Transport_Rate:{transport_rate:.3f}, Exec:{exec_rate:.3f}")

generate_video()
