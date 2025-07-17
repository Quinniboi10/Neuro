#pragma once

#include "types.h"

struct LRSchedule {
	virtual float lr(u64 epoch) = 0;

	virtual ~LRSchedule() = default;
};

namespace lrSchedules {
	struct ConstantLR : LRSchedule {
		float constLR;

		ConstantLR(float constLR) : constLR(constLR) {}

		float lr(u64 epoch) override { return constLR; }
	};
}