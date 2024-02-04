#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 10:32:03 2023

@author: Roger Balsach
"""
from collections import defaultdict
from itertools import count

import sympy as sym


hash_prime = 162_259_276_829_213_363_391_578_010_288_127


class _DummyIndex:
    __instance = {3: None, 8: None}

    def __new__(cls, dim):
        if cls.__instance[dim] is None:
            cls.__instance[dim] = super().__new__(cls)
        return cls.__instance[dim]

    def __init__(self, dim):
        self.dim = dim


class CustomOperation:
    _index_dict = None
    _index = None

    def get_free_indices(self):
        return self._index

    @property
    def index(self):
        return self.get_free_indices()

    def __add__(self, other):
        return Add(self, other)

    def __sub__(self, other):
        return Add(self, -other)

    def __mul__(self, other):
        return Mul(self, other)

    def __rmul__(self, other):
        return Mul(self, other)

    def __neg__(self):
        return Mul(-1, self)

    def __truediv__(self, other):
        return Mul(self, Pow(other, -1))

    def __rtruediv__(self, other):
        return Mul(other, Pow(self, -1))

    def __pow__(self, other, mod=None):
        if mod is None:
            return Pow(self, other)
        return NotImplemented

    def __hash__(self):
        # I will use the convention to combine symmetric hashes by addition,
        # so I don't want the hash to be linear. Also I am modding by a large
        # prime number to keep the hashes small.
        return pow(super().__hash__(), 2, mod=2_305_843_009_213_693_951)


# class One(CustomOperation, sym.core.numbers.One):
#     def __mul__(self, other):
#         return other

#     def __neg__(self):
#         return -1


# class NegativeOne(CustomOperation, sym.core.numbers.NegativeOne):
#     pass


class ImaginaryUnit(CustomOperation, sym.core.numbers.ImaginaryUnit):
    pass


I = ImaginaryUnit()


class Const(CustomOperation, sym.Symbol):
    _index = []


class Add(CustomOperation, sym.Add):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        self._check_indices()

    def _check_indices(self):
        first = self.args[0]
        if hasattr(first, 'index'):
            self._index = self.args[0].index
        else:
            self._index = []
        for arg in self.args:
            if (hasattr(arg, 'index')
                    and (not arg.index or set(arg.index) == set(self._index))):
                continue
            elif self._index == []:
                continue
            raise ValueError('Indices not compatible.')

    def get_free_indices(self):
        if self._index is None:
            self._check_indices()
        return self._index


class Mul(CustomOperation, sym.Mul):
    _has_delta = None
    _has_contracted_delta = None
    _has_adjoint_index = None
    _has_contracted_adjoint_index = None

    def __new__(cls, *args, **kwargs):
        terms = cls.expand_args(args)
        if len(terms) > 1:
            return Add(*(Mul(*term) for term in terms))
        return super().__new__(cls, *args, **kwargs)

    def __call__(self, *args):
        return type(self)(*args)

    @classmethod
    def expand_args(cls, args):
        '''
        Uses distributive property to simplify a products of sums into sums
        of products.
        '''
        terms = [[]]
        for arg in args:
            if isinstance(arg, Add):
                new_terms = []
                exp_subterms = []
                for subterm in arg.args:
                    exp_subterms.extend(cls.expand_args(subterm.args))
                for term in terms:
                    for subterm in exp_subterms:
                        new_terms.append([*term, *subterm])
                terms = new_terms
            elif isinstance(arg, Mul):
                for term in terms:
                    term.extend(arg.args)
            else:
                for term in terms:
                    term.append(arg)
        return tuple(terms)

    def get_free_indices(self):
        if self._index is None:
            free_idx = set()
            for idx, e in self.index_dict.items():
                if len(e) > 1:
                    continue
                free_idx.add(idx)
            self._index = list(free_idx)
        return self._index

    def update_index_dict(self, other):
        for arg in other:
            if not hasattr(arg, 'index') or not arg.index:
                continue
            for idx in arg.index:
                self._index_dict[idx].append(arg)

    @property
    def index_dict(self):
        if self._index_dict is None:
            self._index_dict = defaultdict(list)
            self.update_index_dict(self.args)
        return {k: v.copy() for k, v in self._index_dict.items()}

    def has_delta(self):
        if self._has_delta is None or self._has_contracted_delta is None:
            self._has_delta = []
            self._has_contracted_delta = []
            for arg in self.args:
                if not isinstance(arg, delta):
                    continue
                self._has_delta.append(arg)
                for idx in arg.index:
                    if len(self.index_dict[idx]) > 1:
                        self._has_contracted_delta.append(arg)
                        break
        return self._has_delta

    def has_contracted_delta(self):
        if self._has_contracted_delta is None:
            self.has_delta()
        return self._has_contracted_delta

    def has_adjoint_index(self):
        if (self._has_adjoint_index is None
                or self._has_contracted_adjoint_index is None):
            # Get a list of all adjoint indices
            self._has_adjoint_index = [idx for idx in self.index_dict
                                       if idx.dim == 8]
            # Get a list of all repeated adjoint indices
            self._has_contracted_adjoint_index = [
                idx for idx in self._has_adjoint_index
                if len(self.index_dict[idx]) == 2
            ]
        return self._has_adjoint_index

    def has_contracted_adjoint_index(self):
        if self._has_contracted_adjoint_index is None:
            self.has_adjoint_index()
        return self._has_contracted_adjoint_index

    def __eq__(self, other):
        # TODO: Implement properly
        return hash(self) == hash(other)

    def __hash__(self):
        # TODO: Modify hash to make it accurate. Right now these objects have
        # the same hash:
            # T^a_{ik}T^a_{kl}T^b_{lm}T^b_{mj}
            # T^a_{ik}T^b_{kl}T^a_{lm}T^b_{mj}

        # Encode the object class into the hash
        hash0 = hash(type(self))
        # Hash 1 is the sum of the hashes of all the arguments.
        # The arguments must be dummyfied, i.e. they should forget about
        # contracted indices,
        # such that A_{ik}B_{kj} has the same hash as A_{im}B_{mj}.
        hash1 = 0
        # Contractions store the information about how the dummy indices are
        # contracted.
        contractions = {}
        for arg in self.args:
            if not hasattr(arg, 'index') or not arg.index:
                hash1 += hash(arg)
                continue
            new_idx = []
            for idx in arg.index:
                if idx in self.index:
                    new_idx.append(idx)
                    continue
                new_idx.append(_DummyIndex(abs(idx.dim)))
            dummified_arg = type(arg)(*new_idx)
            hash1 += hash(dummified_arg)
            for idx in arg.index:
                if idx not in self.index:
                    value = contractions.get(idx, [])
                    value.append(dummified_arg)
        hash2 = pow(sum(hash(v) for v in contractions.values()), 2, mod=hash1)
        return pow(hash0 + pow(hash1, 2, mod=hash0) + hash2, 2, mod=hash_prime)

    def __copy__(self):
        return type(self)(*self.args)


class Pow(CustomOperation, sym.Pow):
    pass


class Basis:
    def __init__(self, *args):
        bv = {sym.Symbol(f'c_{i}'): arg for i, arg in enumerate(args)}
        self.basis_vectors = bv
        self.dim = len(args)
        self._check_basis(args)

        for i, arg in enumerate(args):
            self.__setattr__(f'c{i}', arg)

    def _check_basis(self, args):
        self.free_idx = set(args[0].get_free_indices())
        coeffs = []
        std_vec = set()
        for arg in args:
            free_idx = set(arg.get_free_indices())
            if free_idx != self.free_idx:
                raise ValueError(
                    f"Basis vector {arg} has incompatible indices with "
                    + f"{args[0]}."
                )
            # Write the basis vectors in the "standard basis"
            coeffs.append(self.extract_coeff(arg))
            # Find the minimal set of "standard vectors" that span this basis.
            # This is done to make the matrices as small as posible.
            # In general, one will need more than self.dim "standard vectors"
            # to span all the vectors in the basis.
            for vec in coeffs:
                std_vec |= set(vec)
        self.std_vec = list(std_vec)
        # Find the transition matrix "basis <- standard"
        M = sym.zeros(len(args), len(self.std_vec))
        for i, arg in enumerate(coeffs):
            for j, vec in enumerate(self.std_vec):
                if vec in arg:
                    M[i, j] = arg[vec]
        if len(M.rowspace()) < M.rows:
            raise ValueError('The basis vectors must be linearly independent.')
        # Add new elements in the basis to make it invertible.
        for vec in M.nullspace():
            M = M.row_insert(vec.T)
        self.basis_transf = (M.inv()).T

    def extract_coeff(self, arg):
        arg = sunsimplify(arg)
        if isinstance(arg, Add):
            terms = arg.args
        elif isinstance(arg, Mul):
            terms = [arg]
        else:
            raise NotImplementedError(
                f"Don't know how to extract the coefficients of {type(arg)} object."
            )
        coeffs = {}
        for term in terms:
            vector = 1
            coeff = []
            for element in term.args:
                if isinstance(element, (delta, T)):
                    vector *= element
                else:
                    coeff.append(element)
            coeffs[vector] = coeffs.get(vector, 0) + Mul(*coeff)
        return coeffs

    def __call__(self, arg):
        if set(arg.index) != self.free_idx:
            raise ValueError(
                "expresion has incompatible indices with the basis."
            )

        coeffs = self.extract_coeff(arg)
        M = sym.zeros(len(self.std_vec), 1)
        for i, vec in enumerate(self.std_vec):
            M[i] = coeffs.pop(vec, 0)
        if coeffs:
            raise ValueError('expression cannot be expressed in this basis.')
        ci = self.basis_transf * M
        ci, extra = ci[:self.dim, 0], ci[self.dim:, 0]
        if extra and not extra.is_zero:
            raise ValueError('expression is not is the basis span.')
        ci.simplify()
        return (ci.T * sym.Matrix(list(self.basis_vectors)))[0]

    def __repr__(self):
        return repr(self.basis_vectors)


N = Const('N_c')
C = Const('C_F')
Tr = Const('T_F')


class Index(sym.Symbol):
    free_numbers = {3: [], -3: [], 8: []}
    forbiden_numbers = {3: [0], -3: [0], 8: [0]}
    next_number = {3: count(1), -3: count(1), 8: count(1)}

    def __new__(cls, dim=3, symbol=None, *args, **kwargs):
        if symbol is None:
            symbol = cls.get_symbol(dim)
        return super().__new__(cls, symbol, *args, **kwargs)

    def __init__(self, dim=3, symbol=None, *args, **kwargs):
        self.dim = dim
        self.free_numbers = type(self).free_numbers[dim]
        self.fn = self.forbiden_numbers[dim]
        self.objects = set()
        self.custom_symbol = True
        if symbol is None:
            self.custom_symbol = False
        else:
            if symbol[0] in {'i', 'j', 'a'} and symbol[1 % len(symbol)] == '_':
                start, end = 2, None
                if symbol[2] == '{' and symbol[-1] == '}':
                    start, end = 3, -1
                try:
                    d = int(symbol[start:end])
                    if d in self.fn:
                        raise Exception('Name already used')
                    self.fn.append(d)
                    self.custom_symbol = False
                except ValueError:
                    pass
        super().__init__(*args, **kwargs)

    @classmethod
    def get_symbol(cls, dim):
        if dim == 3:
            symbol = 'i'
        elif dim == -3:
            symbol = 'j'
        elif dim == 8:
            symbol = 'a'
        free_numbers = cls.free_numbers[dim]
        next_number = cls.next_number[dim]
        forbiden_numbers = cls.forbiden_numbers[dim]
        number = 0
        while number in forbiden_numbers:
            if free_numbers:
                number = sorted(free_numbers, reverse=True).pop()
            else:
                number = next(next_number)
        forbiden_numbers.append(number)
        return f'{symbol}_{{{number}}}'


class IndexSymbol(CustomOperation, sym.Symbol):
    def set_index(self, i, pos):
        if i.dim != self.index_type[pos]:
            raise ValueError(
                f'Index {i} should have dimension {self.index_type[pos]}, '
                + f'but has has dimension {i.dim}'
            )
        self._index[pos] = i

    @property
    def index(self):
        if self._index is None:
            self.get_free_indices()
        return self._index


class T(IndexSymbol):
    index_type = (8, 3, 3)

    def __new__(cls, a, i, j, *args, **kwargs):
        for n, idx in enumerate((a, i, j)):
            valid_idx = (
                isinstance(idx, _DummyIndex)
                or isinstance(idx, Index) and abs(idx.dim) == cls.index_type[n]
            )
            if not valid_idx:
                raise ValueError(idx)
        symbol = f"T^{{{a}}}_{{{i}{j}}}"
        if i == j:
            return 0
        return super().__new__(cls, name=symbol, *args, **kwargs)

    def __init__(self, a, i, j):
        self._index = (a, i, j)

    @property
    def a(self):
        return self._index[0]

    @property
    def i(self):
        return self._index[1]

    @property
    def j(self):
        return self._index[2]

    def change_idx(self, old, new):
        new_idx = list(self.index)
        pos = new_idx.index(old)
        new_idx[pos] = new
        return type(self)(*new_idx)


class delta(IndexSymbol):
    def __new__(cls, i, j):
        if not (isinstance(i, (Index, _DummyIndex))
                and isinstance(j, (Index, _DummyIndex))):
            raise ValueError(i, j)
        if abs(i.dim) != abs(j.dim):
            raise ValueError()
        if i == j:
            if abs(i.dim) == 3:
                return N
            elif i.dim == 8:
                return N**2 - 1
        if abs(i.dim) == 3:
            op = '_'
        elif i.dim == 8:
            op = '^'
        symbol = f'\delta{op}{{{i}{j}}}'
        return super().__new__(cls, name=symbol)

    def __init__(self, i, j):
        self._index = [0, 0]
        self.index_type = (i.dim, j.dim)
        self.set_index(i, pos=0)
        self.set_index(j, pos=1)

    @property
    def i(self):
        return self._index[0]

    @property
    def a(self):
        return self._index[0]

    @property
    def j(self):
        return self._index[1]

    @property
    def b(self):
        return self._index[1]

    def change_idx(self, old, new):
        new_idx = list(self.index)
        new_idx.remove(old)
        new_idx.append(new)
        return type(self)(*new_idx)

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        if self.i == other.i:
            if self.j == other.j:
                return True
        if self.i == other.j and self.j == other.i:
            return True
        return False

    def __hash__(self):
        # Encode the the class into the hash
        hash0 = hash(type(self))
        # Combine the indices in a symmetric way, so that the hash doesn't
        # distinguis between delta(a,b) and delta(b,a)
        hash1 = sum(hash(idx) for idx in self.index)
        # Combine hashes, raise to the second power to avoid linearity, i.e.
        # hash(delta(i,j)) + hash(delta(k,l)) !=
        # hash(delta(i,l)) + hash(delta(k,j))
        return pow(hash0 + pow(hash1, 2, mod=hash0), 2, mod=hash_prime)


class f(IndexSymbol):
    index_type = (8, 8, 8)

    def __new__(cls, a, b, c, *args, **kwargs):
        for n, idx in enumerate((a, b, c)):
            valid_idx = (
                isinstance(idx, _DummyIndex)
                or isinstance(idx, Index) and abs(idx.dim) == cls.index_type[n]
            )
            if not valid_idx:
                raise ValueError(idx)
        symbol = f"f^{{{a}{b}{c}}}"
        if a == b or a == c or b == c:
            return 0
        return super().__new__(cls, name=symbol, *args, **kwargs)

    def __init__(self, a, b, c):
        self._index = [0, 0, 0]
        self.set_index(a, pos=0)
        self.set_index(b, pos=1)
        self.set_index(c, pos=2)


class d(IndexSymbol):
    index_type = (8, 8, 8)

    def __new__(cls, a, b, c, *args, **kwargs):
        for n, idx in enumerate((a, b, c)):
            valid_idx = (
                isinstance(idx, _DummyIndex)
                or isinstance(idx, Index) and abs(idx.dim) == cls.index_type[n]
            )
            if not valid_idx:
                raise ValueError(idx)
        symbol = f"d^{{{a}{b}{c}}}"
        return super().__new__(cls, name=symbol, *args, **kwargs)

    def __init__(self, a, b, c):
        self._index = [0, 0, 0]
        self.set_index(a, pos=0)
        self.set_index(b, pos=1)
        self.set_index(c, pos=2)


def _contract_deltas(expr):
    new_args = list(expr.args)
    # Look only at deltas with repeated indices
    for arg in expr.has_contracted_delta():
        # Look at both indices of the delta
        for idx in arg.index:
            # Check whether the index is repeated
            if len(other_ob := expr.index_dict[idx]) == 1:
                continue
            # Find the other object with the same index as the delta
            other_ob.remove(arg)
            other_ob = other_ob.pop()
            # Find the other index of the delta
            other_idx = arg.index.copy()
            other_idx.remove(idx)
            other_idx = other_idx.pop()
            # Remove the delta from the expression
            new_args.remove(arg)
            # Set the index of the other object to 'other_idx'
            new_args.remove(other_ob)
            new_args.append(other_ob.change_idx(idx, other_idx))
            return Mul(*new_args)
    return expr


def _remove_rep_a(expr):
    new_args = list(expr.args)
    for idx in expr.has_contracted_adjoint_index():
        ob1, ob2 = expr.index_dict[idx]
        # For the case where both ob1 and ob2 are in the
        # fundamental representation, use the relation
        # T^a_{ij} T^a_{kl}
        # = Tr (delta_{il} delta_{kj} - 1/N delta_{ij} delta_{kl})
        if isinstance(ob1, T) and isinstance(ob2, T):
            new_args.remove(ob1)
            new_args.remove(ob2)
            # Check if a fundamental index is repeated,
            # in that case use the relation
            # T^a_{ij} T^a_{jk} = Cf delta_{ik}
            if ob1.i == ob2.j:
                ob1, ob2 = ob2, ob1
            if ob1.j == ob2.i:
                new_args.extend([C, delta(ob1.i, ob2.j)])
            else:
                new_args.extend(
                    [Tr,
                     (delta(ob1.i, ob2.j) * delta(ob2.i, ob1.j)
                      - 1/N * delta(ob1.i, ob1.j) * delta(ob2.i, ob2.j))]
                )
            return Mul(*new_args)
        if isinstance(ob1, T):
            ob1, ob2 = ob2, ob1
        # Simplify using the relation
        # f^{abc} T^c_{ij} = -2i Tr [T^a, T^b]_{ij}
        if isinstance(ob1, f) and isinstance(ob2, T):
            sign = 1
            other_idx = list(ob1.index)
            if idx == other_idx[1]:
                sign = -1
            other_idx.remove(idx)
            idx1, idx2 = other_idx
            new_args.remove(ob1)
            new_args.remove(ob2)
            k = Index(3)
            new_args.append(-2*I*sign*Tr*(
                T(idx1, ob2.i, k)*T(idx2, k, ob2.j)
                - T(idx2, ob2.i, k)*T(idx1, k, ob2.j)
            ))
            return Mul(*new_args)
        elif isinstance(ob1, d) and isinstance(ob2, T):
            other_idx = list(ob1.index)
            other_idx.remove(idx)
            idx1, idx2 = other_idx
            new_args.remove(ob1)
            new_args.remove(ob2)
            dummy = Index(3)
            new_args.append(
                2*Tr*(T(idx1, ob2.i, dummy)*T(idx2, dummy, ob2.j)
                      + T(idx2, ob2.i, dummy)*T(idx1, dummy, ob2.j))
                - 4*Tr**2/N*delta(idx1, idx2)*delta(ob2.i, ob2.j)
            )
            return Mul(*new_args)
        else:
            raise NotImplementedError()
    return expr


def _remove_traces(expr):
    new_args = list(expr.args)
    Ts = []
    for term in expr.args:
        if not isinstance(term, T):
            continue
        for t in Ts:
            if t.i != term.j or t.j != term.i:
                continue
            new_args.remove(t)
            new_args.remove(term)
            new_args.extend([Tr, delta(t.a, term.a)])
            return Mul(*new_args)
        Ts.append(term)
    return expr


def sunsimplify(expr):
    while isinstance(expr, sym.Mul):
        # First, contract all possible deltas
        if expr.has_contracted_delta():
            expr = _contract_deltas(expr)
            continue
        # Then, remove all repeated adjoint indices
        if expr.has_contracted_adjoint_index():
            expr = _remove_rep_a(expr)
            continue
        expr = _remove_traces(expr)
        return expr
    if hasattr(expr, 'args'):
        return type(expr)(*(sunsimplify(arg) for arg in expr.args))
    return expr


a = Index(8)
b = Index(8)
c = Index(8)
i1 = Index(3, 'i_{1}')
i2 = Index(3, 'i_{2}')
i3 = Index(3)
i4 = Index(3)
j1 = Index(-3, 'j_{1}')
j2 = Index(-3, 'j_{2}')
j3 = Index(-3)
j4 = Index(-3)
k1 = Index(-3)
k2 = Index(3)

i = Index(3)
j = Index(3)
k = Index(3)


c1 = delta(i1, j1)*delta(i2, j2)
assert sunsimplify(c1) == c1
c8 = T(a, i1, j1)*T(a, i2, j2)
assert sunsimplify(c8) == Tr*(delta(i1,j2)*delta(i2,j1) - delta(i1,j1)*delta(i2,j2)/N)

B = Basis(c1, c8)
c0, c1 = list(B.basis_vectors)

C12_1 = -T(a, i1, j3)*delta(j3, i3)*T(a,i3,j1)*delta(i2,j4)*delta(j4,i4)*delta(i4, j2)
assert B(C12_1) == -C*c0
C12_8 = -T(a, i1, j3)*T(b, j3, i3)*T(a,i3,j1)*delta(i2,j4)*T(b, j4, i4)*delta(i4, j2)
assert B(C12_8) == Tr*c1/N

C13_1 = -T(a, i2, i4)*T(a,j3,j1)*delta(i1,j3)*delta(i4,j2)
assert B(C13_1) == -c1
C13_8 = -T(a, i2, i4)*T(a,j3,j1)*T(b,i1,j3)*T(b,i4,j2)
assert sym.simplify(B(C13_8) - (-C*Tr/N*c0 + (Tr/N - C)*c1)) == 0

C14_1 = T(a, j4, j2)*T(a, j3, j1)*delta(i1,j3)*delta(i2,j4)
assert B(C14_1) == c1
C14_8 = T(a, j4, j2)*T(a, j3, j1)*T(b,i1,j3)*T(b,i2,j4)
B(C14_8)

c1 = delta(a, b)*delta(i1, j1)
c8s = d(c,a,b)*T(c,i1,j1)
# x = 2*T(c,i,j)*(T(a,j,k)*T(b,k,i)+T(b,j,k)*T(a,k,i))*T(c,i1,j1)
c8a = I*f(c,a,b)*T(c,i1,j1)
sunsimplify(c8a)
Basis(c1, c8s, c8a)

l = Index(3)
m = Index(3)
x1 = T(a, i, k)*T(a,k,l)*T(b,l,m)*T(b,m,j)
print(hash(x1))
x2 = T(a, i, k)*T(b,k,l)*T(a,l,m)*T(b,m,j)
print(hash(x2))
