"""
This module implements a HallSet class, which is needed in this repository to convert the log-signature of a path to
a Hall basis and identify the corresponding tensor algebra element.
"""

import functools
import itertools
from collections import defaultdict
from typing import Generator, Union

import jax.numpy as jnp
import numpy as np
from scipy.sparse import csc_matrix


def tkey_to_index(width: int, tkey: Union[int, tuple[int]]) -> int:
    if isinstance(tkey, int):
        assert tkey <= width
        return tkey

    result = 0
    for letter in tkey:
        result *= width
        result += letter
    return result


def tensor_algebra_dimension(width: int, depth: int) -> int:
    result = 1
    for _ in range(depth):
        result *= width
        result += 1
    return result


def generate_tensor_keys_level(
    width: int, degree: int
) -> Generator[tuple[int], None, None]:
    if degree == 1:
        yield from ((i,) for i in range(1, width + 1))
        return

    for i in range(1, width + 1):
        yield from ((i, *r) for r in generate_tensor_keys_level(width, degree - 1))


def generate_tensor_keys(width: int, depth: int) -> Generator[tuple[int], None, None]:
    yield ()

    if depth == 0:
        return

    for i in range(1, width + 1):
        yield (i,)

    for degree in range(2, depth + 1):
        yield from generate_tensor_keys_level(width, degree)


class HallSet:
    def __init__(self, width, degree=1):
        self.width = width
        self.degree = 1

        self.data = data = []
        self.reverse_map = reverse_map = {}
        self.degree_ranges = degree_ranges = []
        self.sizes = sizes = []
        self.letters = letters = []
        self.l2k = l2k = {}

        data.append((0, 0))
        degree_ranges.append((0, 1))
        sizes.append(0)

        for letter in range(1, width + 1):
            parents = (0, letter)
            letters.append(letter)
            data.append(parents)
            reverse_map[parents] = letter
            l2k[letter] = letter

        degree_ranges.append((degree_ranges[0][1], len(data)))
        sizes.append(width)

        if degree > self.degree:
            self.grow_up(degree)

    def grow_up(self, degree):

        data = self.data
        reverse_map = self.reverse_map
        degree_ranges = self.degree_ranges

        while self.degree < degree:
            next_degree = self.degree + 1
            left = 1
            while 2 * left <= next_degree:
                right = next_degree - left

                ilower, iupper = degree_ranges[left]
                jlower, jupper = degree_ranges[right]

                i = ilower

                while i < iupper:
                    j = max(jlower, i + 1)
                    while j < jupper:
                        if data[j][0] <= i:
                            parents = (i, j)
                            data.append(parents)
                            reverse_map[parents] = len(data) - 1
                        j += 1
                    i += 1
                left += 1

            degree_ranges.append((degree_ranges[-1][1], len(data)))
            self.sizes.append(len(data))
            self.degree += 1

    @functools.lru_cache
    def key_to_string(self, key: int) -> str:
        assert key < len(self.data)

        left, right = self.data[key]

        if left == 0:
            return f"{right}"

        return f"[{self.key_to_string(left)}, {self.key_to_string(right)}]"

    @functools.lru_cache
    def product(self, lhs_key: int, rhs_key: int) -> list[tuple[int, int]]:
        if rhs_key < lhs_key:
            return [(k, -c) for k, c in self.product(rhs_key, lhs_key)]

        if lhs_key == rhs_key:
            return []

        if key := self.reverse_map.get((lhs_key, rhs_key)):
            return [(key, 1)]

        lparent, rparent = self.data[rhs_key]

        left_result = [
            (k, c1 * c)
            for (k1, c1) in self.product(lhs_key, lparent)
            for (k, c) in self.product(k1, rparent)
        ]
        right_result = [
            (k, -c1 * c)
            for (k1, c1) in self.product(lhs_key, rparent)
            for (k, c) in self.product(k1, lparent)
        ]
        result = defaultdict(lambda: 0)
        for k, c in left_result:
            result[k] += c
        for k, c in right_result:
            result[k] += c

        return list(result.items())

    @functools.lru_cache
    def expand(self, key: int) -> list[tuple[int, tuple[int]]]:
        if key in self.letters:
            return [((key,), 1)]

        assert key < len(self.data)
        lparent, rparent = self.data[key]

        left_expansion = self.expand(lparent)
        right_expansion = self.expand(rparent)

        left_terms = [
            ((*k1, *k2), c1 * c2)
            for (k1, c1), (k2, c2) in itertools.product(left_expansion, right_expansion)
        ]
        right_terms = [
            ((*k1, *k2), c1 * c2)
            for (k1, c1), (k2, c2) in itertools.product(right_expansion, left_expansion)
        ]

        result = defaultdict(lambda: 0)
        for k, c in left_terms:
            result[k] += c
        for k, c in right_terms:
            result[k] -= c

        return list(result.items())

    @functools.lru_cache
    def rbracket(self, tkey: Union[int, tuple[int]]) -> list[tuple[int, int]]:
        if isinstance(tkey, int):
            return [(tkey, 1)]

        if len(tkey) == 0:
            return []

        if len(tkey) == 1:
            return [(tkey[0], 1)]

        assert len(tkey) > 1, f"{tkey}"
        first, *remaining = tkey
        return [
            (k, c1 * c)
            for (k1, c1) in self.rbracket(tuple(remaining))
            for k, c in self.product(first, k1)
        ]

    def l2t_matrix(self, degree=None, dtype=np.float32) -> jnp.ndarray:
        degree = degree or self.degree
        tensor_alg_size = tensor_algebra_dimension(self.width, degree)

        indptr = [0, 0]
        indices = []
        data = []
        for lkey in range(1, self.sizes[degree]):
            for k, c in self.expand(lkey):
                indices.append(tkey_to_index(self.width, k))
                data.append(c)
            indptr.append(indptr[-1] + len(self.expand(lkey)))

        data = np.array(data, dtype=dtype)
        indices = np.array(indices, dtype=np.int64)
        indptr = np.array(indptr, dtype=np.int64)
        return jnp.array(
            csc_matrix(
                (data, indices, indptr),
                shape=(tensor_alg_size, self.sizes[degree]),
                dtype=dtype,
            ).toarray()
        )

    def t2l_matrix(self, degree=None, dtype=np.float32) -> jnp.ndarray:
        degree = degree or self.degree
        tensor_alg_size = tensor_algebra_dimension(self.width, degree)

        indptr = [0]
        indices = []
        data = []
        for tkey in generate_tensor_keys(self.width, degree):
            for k, c in self.rbracket(tkey):
                indices.append(k)
                data.append(c / len(tkey))
            indptr.append(len(data))
        data = np.array(data, dtype=dtype)
        indices = np.array(indices, dtype=np.int64)
        indptr = np.array(indptr, dtype=np.int64)

        return jnp.array(
            csc_matrix(
                (data, indices, indptr),
                shape=(self.sizes[degree], tensor_alg_size),
                dtype=dtype,
            ).toarray()
        )
