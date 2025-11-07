# Checkpoint Save Flow - Before and After Fix

## Before Fix (Problem Flow)

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Checkpoint Save Initiated                                │
│    - Create unfinished marker: checkpoint-unfinished        │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Async Save Scheduled                                     │
│    - Checkpoint data written asynchronously                 │
│    - Finalize callback registered                           │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Finalize Callback Executed (PROBLEM HERE)                │
│    ❌ If ANY error occurs:                                  │
│       - Logger notification fails                           │
│       - Distributed barrier fails                           │
│       - Callback group notification fails                   │
│    ❌ Marker removal is SKIPPED                             │
│    ❌ Exception propagates, callback exits                  │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Result: UNFINISHED CHECKPOINT                            │
│    ❌ Marker file remains: checkpoint-unfinished            │
│    ❌ Checkpoint appears incomplete                         │
│    ❌ May be deleted on next cleanup                        │
└─────────────────────────────────────────────────────────────┘
```

## After Fix (Corrected Flow)

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Checkpoint Save Initiated                                │
│    - Create unfinished marker: checkpoint-unfinished        │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Setup Phase (NEW)                                        │
│    ✅ Check if AsyncFinalizerCallback is registered         │
│    ✅ Auto-add if missing                                   │
│    ✅ Log warning if auto-added                             │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Async Save Scheduled                                     │
│    - Checkpoint data written asynchronously                 │
│    - Finalize callback registered                           │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Finalize Callback Executed (ENHANCED)                    │
│    ✅ Try: Notify loggers                                   │
│       └─ Catch: Log warning, continue                       │
│    ✅ Try: Remove marker with barrier                       │
│       └─ Catch: Try without barrier (fallback)              │
│          └─ Catch: Log error, continue                      │
│    ✅ Try: Notify callback group                            │
│       └─ Catch: Log warning, continue                       │
│    ✅ Try: Create checkpoint links                          │
│       └─ Catch: Log warning, continue                       │
│    ✅ Try: Remove deferred checkpoints                      │
│       └─ Catch: Log warning, continue                       │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. Marker Removal (ENHANCED)                                │
│    ✅ Try: Remove marker file                               │
│       └─ Catch: Wait 0.1s, retry once                       │
│          └─ Catch: Log error (marker removal failed)        │
│    ✅ Detailed logging at each step                         │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. Training End (NEW)                                       │
│    ✅ Check for pending async saves                         │
│    ✅ Finalize all pending saves (blocking)                 │
│    ✅ Log number of saves finalized                         │
│    ✅ Ensure no unfinished markers remain                   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 7. Result: COMPLETE CHECKPOINT                              │
│    ✅ Marker file removed: checkpoint-unfinished (deleted)  │
│    ✅ Checkpoint is complete and valid                      │
│    ✅ Ready for use/resumption                              │
└─────────────────────────────────────────────────────────────┘
```

## Key Improvements

### 1. Error Isolation
```
Before: Error in step A → Entire callback fails → Marker not removed
After:  Error in step A → Log warning → Continue to marker removal
```

### 2. Fallback Mechanisms
```
Primary:   Remove marker with distributed barrier
Fallback:  Remove marker without barrier (if distributed unavailable)
Retry:     Wait 0.1s and try again (if filesystem busy)
```

### 3. Guaranteed Cleanup
```
During Training:  AsyncFinalizerCallback finalizes on batch/epoch end
On Training End:  Explicit finalization of all pending saves
On Next Run:      Automatic cleanup of any remaining unfinished markers
```

## Error Handling Strategy

```
┌──────────────────────────────────────────────────────────┐
│ Operation Type          │ Error Handling                 │
├─────────────────────────┼────────────────────────────────┤
│ Logger Notification     │ Log warning, continue          │
│ Callback Group Notify   │ Log warning, continue          │
│ Checkpoint Linking      │ Log warning, continue          │
│ Deferred Removal        │ Log warning per checkpoint     │
│ Marker Removal          │ Retry once, log error          │
│ Distributed Barrier     │ Fallback to no-barrier mode    │
└──────────────────────────────────────────────────────────┘
```

## Monitoring Points

```
Setup Phase:
  ⚠️  "async_save is enabled but AsyncFinalizerCallback is not registered"
  
Finalization:
  ❌  "Failed to remove unfinished checkpoint marker"
  ✅  "Successfully removed unfinished marker"
  ✅  "Async checkpoint save finalized successfully"
  
Training End:
  ℹ️  "Finalizing N pending async checkpoint save(s)"
  ✅  "All pending async checkpoint saves finalized successfully"
  
Marker Removal:
  ⚠️  "Failed to remove unfinished marker" (first attempt)
  ✅  "Successfully removed unfinished marker on retry"
  ❌  "Failed to remove unfinished marker on retry" (both attempts failed)
```

## Recovery Scenarios

### Scenario 1: Training Interrupted
```
Before Fix:
  Training stops → Async save incomplete → Marker remains → ❌ Unfinished checkpoint

After Fix:
  Training stops → Next run detects marker → Auto-cleanup → ✅ Clean state
```

### Scenario 2: Distributed Failure
```
Before Fix:
  Barrier fails → Callback exits → Marker remains → ❌ Unfinished checkpoint

After Fix:
  Barrier fails → Try without barrier → Marker removed → ✅ Complete checkpoint
```

### Scenario 3: Filesystem Busy
```
Before Fix:
  Unlink fails → Exception raised → Marker remains → ❌ Unfinished checkpoint

After Fix:
  Unlink fails → Wait 0.1s → Retry → Marker removed → ✅ Complete checkpoint
```

### Scenario 4: Missing Callback
```
Before Fix:
  No AsyncFinalizerCallback → Saves never finalized → ❌ All checkpoints unfinished

After Fix:
  Setup detects missing callback → Auto-add → Saves finalized → ✅ All checkpoints complete
```

## Testing Coverage

```
✅ Marker creation and removal
✅ Distributed checkpoint markers
✅ EMA checkpoint markers (shared with base)
✅ Syntax validation
✅ Error handling paths
✅ Retry logic
✅ Fallback mechanisms
```
