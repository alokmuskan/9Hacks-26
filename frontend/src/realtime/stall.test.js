import { describe, expect, it } from "vitest";
import { isStreamStalledByMetadata } from "./stall";

describe("isStreamStalledByMetadata", () => {
  it("returns false when pipeline is not running", () => {
    expect(
      isStreamStalledByMetadata({
        running: false,
        nowMs: 10_000,
        lastEventTsMs: 0,
        lastSequence: 0,
        lastSequenceChangeTsMs: 0,
        lastFrameUtcTsMs: 0
      })
    ).toBe(false);
  });

  it("returns true when events stop arriving", () => {
    expect(
      isStreamStalledByMetadata({
        running: true,
        nowMs: 10_000,
        lastEventTsMs: 6_000,
        lastSequence: 12,
        lastSequenceChangeTsMs: 9_500,
        lastFrameUtcTsMs: 9_500
      })
    ).toBe(true);
  });

  it("returns true when sequence is not changing", () => {
    expect(
      isStreamStalledByMetadata({
        running: true,
        nowMs: 10_000,
        lastEventTsMs: 9_500,
        lastSequence: 12,
        lastSequenceChangeTsMs: 6_500,
        lastFrameUtcTsMs: 9_500
      })
    ).toBe(true);
  });

  it("returns true when frame timestamp is stale", () => {
    expect(
      isStreamStalledByMetadata({
        running: true,
        nowMs: 10_000,
        lastEventTsMs: 9_500,
        lastSequence: 12,
        lastSequenceChangeTsMs: 9_500,
        lastFrameUtcTsMs: 4_500
      })
    ).toBe(true);
  });

  it("returns false for healthy stream metadata", () => {
    expect(
      isStreamStalledByMetadata({
        running: true,
        nowMs: 10_000,
        lastEventTsMs: 8_500,
        lastSequence: 12,
        lastSequenceChangeTsMs: 8_500,
        lastFrameUtcTsMs: 8_500
      })
    ).toBe(false);
  });
});
