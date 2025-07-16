#pragma once

#include "types.h"

#include <algorithm>
#include <sstream>
#include <cmath>

#ifdef _WIN32
#include <windows.h>
#include <fcntl.h>
#include <io.h>
#endif

struct UnicodeTerminalInitializer {
	UnicodeTerminalInitializer() {
#ifdef _WIN32
		SetConsoleOutputCP(CP_UTF8);
#endif
	}
};

static inline UnicodeTerminalInitializer unicodeTerminalInitializer;

// Recursively fills a vector with a given value
template<typename T, typename U>
inline void deepFill(T& dest, const U& val) {
	dest = val;
}

template<typename T, typename U>
inline void deepFill(vector<T>& arr, const U& value) {
	for (auto& element : arr) {
		deepFill(element, value);
	}
}

// Formats a number with commas
inline string formatNum(i64 v) {
	auto s = std::to_string(v);

	int n = s.length() - 3;
	if (v < 0)
		n--;
	while (n > 0) {
		s.insert(n, ",");
		n -= 3;
	}

	return s;
}