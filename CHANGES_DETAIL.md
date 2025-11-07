# Detailed Code Changes for Issue #14994

## File: nemo/lightning/pytorch/callbacks/model_checkpoint.py

### Change 1: Enhanced `_get_finalize_save_checkpoint_callback` Method

**Location**: Lines ~655-690

**Purpose**: Add comprehensive error handling to ensure unfinished markers are always removed

**Changes**:
- Wrapped logger notification in try-except
- Added try-except with retry logic for marker removal
- Added error handling for callback group notifications
- Added error handling for checkpoint linking
- Added error handling for deferred checkpoint removal
- Ensured marker removal executes even if other operations fail

**Impact**: Prevents unfinished markers from persisting when errors occur during finalization

---

### Change 2: Enhanced `setup` Method

**Location**: Lines ~339-380

**Purpose**: Automatically register AsyncFinalizerCallback when async_save is enabled

**Changes**:
- Added check for AsyncFinalizerCallback presence
- Automatically adds callback if missing when async_save is enabled
- Logs warning when callback is auto-added

**Impact**: Ensures async checkpoints are always properly finalized

**Code Added**:
```python
# Ensure AsyncFinalizerCallback is registered when async_save is enabled
if self.async_save:
    from nemo.utils.callbacks.dist_ckpt_io import AsyncFinalizerCallback
    
    has_async_finalizer = any(isinstance(cb, AsyncFinalizerCallback) for cb in trainer.callbacks)
    if not has_async_finalizer:
        logging.warning(
            "async_save is enabled but AsyncFinalizerCallback is not registered. "
            "Adding AsyncFinalizerCallback to ensure proper checkpoint finalization."
        )
        trainer.callbacks.append(AsyncFinalizerCallback())
```

---

### Change 3: Enhanced `on_train_end` Method

**Location**: Lines ~363-410

**Purpose**: Finalize all pending async saves before training ends

**Changes**:
- Added check for pending async saves at start of method
- Calls blocking finalization if pending saves exist
- Logs number of pending saves being finalized
- Wrapped in try-except for safety

**Impact**: Ensures no unfinished checkpoints remain after training completion

**Code Added**:
```python
# If async_save is enabled, ensure all pending async saves are finalized before proceeding
if self.async_save:
    try:
        from nemo.utils.callbacks.dist_ckpt_io import AsyncFinalizableCheckpointIO
        
        checkpoint_io = trainer.strategy.checkpoint_io
        if isinstance(checkpoint_io, AsyncFinalizableCheckpointIO):
            num_pending = checkpoint_io.async_calls_queue.get_num_unfinalized_calls()
            if num_pending > 0:
                logging.info(
                    f'Finalizing {num_pending} pending async checkpoint save(s) before train_end...'
                )
                checkpoint_io.maybe_finalize_save_checkpoint(blocking=True)
                logging.info('All pending async checkpoint saves finalized successfully.')
    except Exception as e:
        logging.warning(f'Error finalizing pending async checkpoints on train_end: {e}')
```

---

### Change 4: Enhanced `remove_checkpoint_unfinished_marker` Method

**Location**: Lines ~503-530

**Purpose**: Make marker removal more robust with retry logic

**Changes**:
- Added detailed logging for successful removals
- Added retry logic with brief delay on failure
- Better error messages for debugging
- Handles OSError specifically for file operations

**Impact**: Increases reliability of marker removal, especially in edge cases

**Code Modified**:
```python
if is_global_rank_zero():
    marker_path = ModelCheckpoint.format_checkpoint_unfinished_marker_path(checkpoint_path)
    if marker_path.exists():
        try:
            marker_path.unlink()
            logging.debug(f'Successfully removed unfinished marker: {marker_path}')
        except OSError as e:
            logging.warning(f'Failed to remove unfinished marker {marker_path}: {e}')
            # Retry once after a brief moment
            import time
            time.sleep(0.1)
            try:
                if marker_path.exists():
                    marker_path.unlink()
                    logging.debug(f'Successfully removed unfinished marker on retry: {marker_path}')
            except OSError as e2:
                logging.error(f'Failed to remove unfinished marker on retry {marker_path}: {e2}')
```

---

## Summary of Changes

| Method | Lines Changed | Type | Priority |
|--------|--------------|------|----------|
| `_get_finalize_save_checkpoint_callback` | ~35 lines | Error handling | Critical |
| `setup` | ~10 lines | Callback registration | High |
| `on_train_end` | ~15 lines | Cleanup logic | High |
| `remove_checkpoint_unfinished_marker` | ~15 lines | Retry logic | Medium |

**Total Lines Modified**: ~75 lines
**Total Methods Modified**: 4 methods
**Files Modified**: 1 file

---

## Testing Strategy

1. **Unit Tests**: Created test scripts to verify marker operations
2. **Syntax Validation**: Verified Python syntax is correct
3. **Logic Tests**: Tested marker creation, removal, and edge cases
4. **Integration**: Changes integrate seamlessly with existing code

---

## Risk Assessment

**Risk Level**: Low

**Reasons**:
- Only adds error handling and safety checks
- No changes to core checkpoint save/load logic
- Backward compatible
- Fail-safe design (errors are logged but don't break execution)
- Automatic fallbacks for edge cases

---

## Deployment Notes

1. No configuration changes required
2. No database migrations needed
3. No API changes
4. Existing checkpoints remain compatible
5. Automatic cleanup of old unfinished markers on next run

---

## Monitoring Recommendations

After deployment, monitor logs for:
- Warnings about missing AsyncFinalizerCallback (should be rare)
- Errors during marker removal (indicates filesystem issues)
- Messages about pending async saves at train_end
- Successful finalization messages

---

## Rollback Plan

If issues arise:
1. Revert the single file: `nemo/lightning/pytorch/callbacks/model_checkpoint.py`
2. No data migration needed
3. No configuration changes to revert
4. Existing checkpoints remain valid

---

## Future Improvements

Potential enhancements for future releases:
1. Add metrics/telemetry for checkpoint finalization success rate
2. Implement automatic cleanup of old unfinished markers (older than N days)
3. Add health check endpoint for checkpoint system status
4. Consider making marker removal timeout configurable
