export function isStreamStalledByMetadata({
  running,
  nowMs,
  lastEventTsMs,
  lastSequence,
  lastSequenceChangeTsMs,
  lastFrameUtcTsMs,
  eventStallMs = 3000,
  sequenceStallMs = 3000,
  frameTimestampStallMs = 5000
}) {
  if (!running) {
    return false;
  }
  const eventAgeMs = nowMs - Number(lastEventTsMs || 0);
  const seqStalled =
    Number(lastSequence || 0) > 0 &&
    nowMs - Number(lastSequenceChangeTsMs || 0) > Number(sequenceStallMs || 3000);
  const frameTsStalled =
    Number(lastFrameUtcTsMs || 0) > 0 &&
    nowMs - Number(lastFrameUtcTsMs || 0) > Number(frameTimestampStallMs || 5000);

  return eventAgeMs > Number(eventStallMs || 3000) || seqStalled || frameTsStalled;
}
