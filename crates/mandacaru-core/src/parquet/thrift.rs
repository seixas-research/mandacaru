//! Thrift compact protocol: a bounds-checked reader and a writer.
//!
//! The reader never trusts a length: every list, map and binary size is
//! checked against the bytes that remain, nesting is capped at
//! `MAX_THRIFT_DEPTH`, and every container it builds is charged against a
//! memory bound proportional to the input length, so a hostile footer cannot
//! make the parse allocate more than `parse_memory_bound(len)` bytes.
//! The writer emits fields in strictly ascending id order.

use crate::parquet::{invalid, MAX_THRIFT_DEPTH};
use crate::Result;

/// Bytes a parse of `input_len` bytes of Thrift may allocate at most.
pub const fn parse_memory_bound(input_len: usize) -> usize {
    input_len.saturating_mul(16).saturating_add(4096)
}

/// Compact protocol wire types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Type {
    Stop,
    BoolTrue,
    BoolFalse,
    I8,
    I16,
    I32,
    I64,
    Double,
    Binary,
    List,
    Set,
    Map,
    Struct,
}

impl Type {
    pub fn code(self) -> u8 {
        match self {
            Self::Stop => 0,
            Self::BoolTrue => 1,
            Self::BoolFalse => 2,
            Self::I8 => 3,
            Self::I16 => 4,
            Self::I32 => 5,
            Self::I64 => 6,
            Self::Double => 7,
            Self::Binary => 8,
            Self::List => 9,
            Self::Set => 10,
            Self::Map => 11,
            Self::Struct => 12,
        }
    }

    pub fn from_code(code: u8) -> Result<Self> {
        Ok(match code {
            0 => Self::Stop,
            1 => Self::BoolTrue,
            2 => Self::BoolFalse,
            3 => Self::I8,
            4 => Self::I16,
            5 => Self::I32,
            6 => Self::I64,
            7 => Self::Double,
            8 => Self::Binary,
            9 => Self::List,
            10 => Self::Set,
            11 => Self::Map,
            12 => Self::Struct,
            other => return Err(invalid(format!("thrift type code {other} is not valid"))),
        })
    }

    fn is_bool(self) -> bool {
        matches!(self, Self::BoolTrue | Self::BoolFalse)
    }
}

/// Bounds-checked compact protocol reader over a byte slice.
pub struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
    depth: usize,
    charged: usize,
    limit: usize,
}

impl<'a> Reader<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Self::with_limit(buf, parse_memory_bound(buf.len()))
    }

    pub fn with_limit(buf: &'a [u8], limit: usize) -> Self {
        Self {
            buf,
            pos: 0,
            depth: 0,
            charged: 0,
            limit,
        }
    }

    pub fn position(&self) -> usize {
        self.pos
    }

    pub fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }

    /// Bytes charged so far against the memory bound.
    pub fn charged(&self) -> usize {
        self.charged
    }

    /// Account for `bytes` about to be allocated for parsed data.
    pub fn charge(&mut self, bytes: usize) -> Result<()> {
        let next = self
            .charged
            .checked_add(bytes)
            .filter(|&n| n <= self.limit)
            .ok_or_else(|| invalid("thrift data expands beyond its memory bound"))?;
        self.charged = next;
        Ok(())
    }

    /// Fail when bytes remain, for a buffer that must be exactly one struct.
    pub fn finish(&self) -> Result<()> {
        if self.remaining() != 0 {
            return Err(invalid(format!(
                "{} unexpected trailing bytes after thrift struct",
                self.remaining()
            )));
        }
        Ok(())
    }

    fn truncated() -> crate::Error {
        invalid("thrift data is truncated")
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if n > self.remaining() {
            return Err(Self::truncated());
        }
        let out = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(out)
    }

    fn byte(&mut self) -> Result<u8> {
        let b = *self.buf.get(self.pos).ok_or_else(Self::truncated)?;
        self.pos += 1;
        Ok(b)
    }

    fn varint(&mut self) -> Result<u64> {
        let mut value = 0u64;
        for shift in (0..70).step_by(7) {
            let b = self.byte()?;
            let low = u64::from(b & 0x7f);
            if shift == 63 && low > 1 {
                return Err(invalid("thrift varint overflows 64 bits"));
            }
            value |= low << shift;
            if b & 0x80 == 0 {
                return Ok(value);
            }
        }
        Err(invalid("thrift varint is longer than 10 bytes"))
    }

    fn enter(&mut self) -> Result<()> {
        if self.depth >= MAX_THRIFT_DEPTH {
            return Err(invalid(format!(
                "thrift nesting is deeper than {MAX_THRIFT_DEPTH}"
            )));
        }
        self.depth += 1;
        Ok(())
    }

    fn leave(&mut self) {
        self.depth -= 1;
    }

    fn expect(ty: Type, want: Type, what: &str) -> Result<()> {
        if ty == want {
            Ok(())
        } else {
            Err(invalid(format!(
                "thrift field is {ty:?} where {what} ({want:?}) is required"
            )))
        }
    }

    /// Value of a boolean field (carried in the field header type) or of a
    /// boolean container element (`ty` is the element type read by the caller
    /// through `bool_element`).
    pub fn boolean(&mut self, ty: Type) -> Result<bool> {
        match ty {
            Type::BoolTrue => Ok(true),
            Type::BoolFalse => Ok(false),
            other => Err(invalid(format!(
                "thrift field is {other:?} where a bool is required"
            ))),
        }
    }

    fn bool_element(&mut self) -> Result<bool> {
        match self.byte()? {
            1 => Ok(true),
            0 | 2 => Ok(false),
            other => Err(invalid(format!(
                "thrift bool element byte {other} is not valid"
            ))),
        }
    }

    pub fn i8(&mut self, ty: Type) -> Result<i8> {
        Self::expect(ty, Type::I8, "an i8")?;
        Ok(self.byte()? as i8)
    }

    pub fn i16(&mut self, ty: Type) -> Result<i16> {
        Self::expect(ty, Type::I16, "an i16")?;
        let v = unzigzag(self.varint()?);
        i16::try_from(v).map_err(|_| invalid("thrift i16 is out of range"))
    }

    pub fn i32(&mut self, ty: Type) -> Result<i32> {
        Self::expect(ty, Type::I32, "an i32")?;
        let v = unzigzag(self.varint()?);
        i32::try_from(v).map_err(|_| invalid("thrift i32 is out of range"))
    }

    pub fn i64(&mut self, ty: Type) -> Result<i64> {
        Self::expect(ty, Type::I64, "an i64")?;
        Ok(unzigzag(self.varint()?))
    }

    pub fn double(&mut self, ty: Type) -> Result<f64> {
        Self::expect(ty, Type::Double, "a double")?;
        let bytes: [u8; 8] = self.take(8)?.try_into().expect("8 bytes");
        Ok(f64::from_le_bytes(bytes))
    }

    /// Length-prefixed bytes, borrowed from the input.
    pub fn binary(&mut self, ty: Type) -> Result<&'a [u8]> {
        Self::expect(ty, Type::Binary, "a binary")?;
        let len = self.varint()?;
        let len = usize::try_from(len).map_err(|_| Self::truncated())?;
        self.take(len)
    }

    /// A UTF-8 string, charged against the memory bound.
    pub fn string(&mut self, ty: Type) -> Result<String> {
        let bytes = self.binary(ty)?;
        let text =
            std::str::from_utf8(bytes).map_err(|_| invalid("thrift string is not valid UTF-8"))?;
        self.charge(text.len())?;
        Ok(text.to_owned())
    }

    /// Header of the next field of a struct, or `None` at the stop byte.
    /// `last` is the previous field id of this struct (0 at the start) and is
    /// updated to the returned id.
    pub fn field_header(&mut self, last: &mut i16) -> Result<Option<(i16, Type)>> {
        let b = self.byte()?;
        if b == 0 {
            return Ok(None);
        }
        let ty = Type::from_code(b & 0x0f)?;
        if ty == Type::Stop {
            return Err(invalid("thrift field header has a stop type with a delta"));
        }
        let delta = i16::from(b >> 4);
        let id = if delta == 0 {
            let v = unzigzag(self.varint()?);
            i16::try_from(v).map_err(|_| invalid("thrift field id is out of range"))?
        } else {
            last.checked_add(delta)
                .ok_or_else(|| invalid("thrift field id overflows"))?
        };
        *last = id;
        Ok(Some((id, ty)))
    }

    /// Read one struct: `f(reader, field_id, field_type)` consumes the field
    /// and returns `true`, or returns `false` to have the field skipped.
    pub fn read_struct(
        &mut self,
        mut f: impl FnMut(&mut Self, i16, Type) -> Result<bool>,
    ) -> Result<()> {
        self.enter()?;
        let mut last = 0i16;
        while let Some((id, ty)) = self.field_header(&mut last)? {
            if !f(self, id, ty)? {
                self.skip(ty)?;
            }
        }
        self.leave();
        Ok(())
    }

    /// Header of a list or set: element type and a count already checked
    /// against the remaining bytes. Does not consume the elements.
    pub fn list_header(&mut self, ty: Type) -> Result<(Type, usize)> {
        if ty != Type::List && ty != Type::Set {
            return Err(invalid(format!(
                "thrift field is {ty:?} where a list is required"
            )));
        }
        let b = self.byte()?;
        let elem = Type::from_code(b & 0x0f)?;
        let mut size = usize::from(b >> 4);
        if size == 15 {
            size = usize::try_from(self.varint()?).map_err(|_| Self::truncated())?;
        }
        // Some writers (fastparquet among them) emit an empty list as the single
        // byte 0x00, i.e. element type 0; accept that only when it is empty.
        if elem == Type::Stop && size != 0 {
            return Err(invalid("thrift list has a stop element type"));
        }
        if size > self.remaining() {
            return Err(Self::truncated());
        }
        Ok((elem, size))
    }

    /// Header of a map: `None` for an empty map, else key and value types and
    /// the entry count (checked against the remaining bytes).
    pub fn map_header(&mut self, ty: Type) -> Result<Option<(Type, Type, usize)>> {
        Self::expect(ty, Type::Map, "a map")?;
        let size = usize::try_from(self.varint()?).map_err(|_| Self::truncated())?;
        if size == 0 {
            return Ok(None);
        }
        let b = self.byte()?;
        let key = Type::from_code(b >> 4)?;
        let value = Type::from_code(b & 0x0f)?;
        if key == Type::Stop || value == Type::Stop {
            return Err(invalid("thrift map has a stop element type"));
        }
        if size > self.remaining() / 2 {
            return Err(Self::truncated());
        }
        Ok(Some((key, value, size)))
    }

    /// Read a whole list, charging `size_of::<T>()` per element before the
    /// vector is allocated. `f` reads one element of the declared type.
    pub fn list<T>(
        &mut self,
        ty: Type,
        mut f: impl FnMut(&mut Self, Type) -> Result<T>,
    ) -> Result<Vec<T>> {
        let (elem, size) = self.list_header(ty)?;
        self.charge(size.saturating_mul(std::mem::size_of::<T>().max(1)))?;
        self.enter()?;
        let mut out = Vec::with_capacity(size);
        for _ in 0..size {
            out.push(f(self, elem)?);
        }
        self.leave();
        Ok(out)
    }

    /// List of i32.
    pub fn list_i32(&mut self, ty: Type) -> Result<Vec<i32>> {
        self.list(ty, |r, t| r.i32(t))
    }

    /// List of strings.
    pub fn list_string(&mut self, ty: Type) -> Result<Vec<String>> {
        self.list(ty, |r, t| r.string(t))
    }

    /// Skip a value of the given type that appears as a struct field.
    pub fn skip(&mut self, ty: Type) -> Result<()> {
        match ty {
            Type::Stop => Err(invalid("thrift cannot skip a stop marker")),
            Type::BoolTrue | Type::BoolFalse => Ok(()),
            _ => self.skip_value(ty),
        }
    }

    /// Skip one container element (a bool element occupies one byte).
    fn skip_element(&mut self, ty: Type) -> Result<()> {
        if ty.is_bool() {
            self.bool_element().map(|_| ())
        } else {
            self.skip_value(ty)
        }
    }

    fn skip_value(&mut self, ty: Type) -> Result<()> {
        match ty {
            Type::Stop => Err(invalid("thrift cannot skip a stop marker")),
            Type::BoolTrue | Type::BoolFalse => self.bool_element().map(|_| ()),
            Type::I8 => self.take(1).map(|_| ()),
            Type::I16 | Type::I32 | Type::I64 => self.varint().map(|_| ()),
            Type::Double => self.take(8).map(|_| ()),
            Type::Binary => {
                let len = usize::try_from(self.varint()?).map_err(|_| Self::truncated())?;
                self.take(len).map(|_| ())
            }
            Type::List | Type::Set => {
                let (elem, size) = self.list_header(ty)?;
                self.enter()?;
                for _ in 0..size {
                    self.skip_element(elem)?;
                }
                self.leave();
                Ok(())
            }
            Type::Map => {
                if let Some((key, value, size)) = self.map_header(ty)? {
                    self.enter()?;
                    for _ in 0..size {
                        self.skip_element(key)?;
                        self.skip_element(value)?;
                    }
                    self.leave();
                }
                Ok(())
            }
            Type::Struct => self.read_struct(|_, _, _| Ok(false)),
        }
    }
}

fn unzigzag(v: u64) -> i64 {
    ((v >> 1) as i64) ^ -((v & 1) as i64)
}

fn zigzag(v: i64) -> u64 {
    ((v << 1) ^ (v >> 63)) as u64
}

/// Compact protocol writer. Struct fields must be written in strictly
/// ascending id order; `field` fails otherwise.
#[derive(Default)]
pub struct Writer {
    buf: Vec<u8>,
    stack: Vec<i16>,
    last: i16,
}

impl Writer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn finish(self) -> Result<Vec<u8>> {
        if !self.stack.is_empty() {
            return Err(invalid("thrift writer finished inside a struct"));
        }
        Ok(self.buf)
    }

    pub fn len(&self) -> usize {
        self.buf.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    fn varint(&mut self, mut v: u64) {
        while v >= 0x80 {
            self.buf.push((v as u8 & 0x7f) | 0x80);
            v >>= 7;
        }
        self.buf.push(v as u8);
    }

    pub fn struct_begin(&mut self) {
        self.stack.push(self.last);
        self.last = 0;
    }

    pub fn struct_end(&mut self) -> Result<()> {
        let outer = self
            .stack
            .pop()
            .ok_or_else(|| invalid("thrift writer closed a struct that was not open"))?;
        self.buf.push(0);
        self.last = outer;
        Ok(())
    }

    /// Field header. For `Type::BoolTrue`/`BoolFalse` the value is the type.
    pub fn field(&mut self, id: i16, ty: Type) -> Result<()> {
        if id <= self.last {
            return Err(invalid(format!(
                "thrift writer: field {id} is not above the previous field {}",
                self.last
            )));
        }
        let delta = id - self.last;
        if (1..=15).contains(&delta) {
            self.buf.push(((delta as u8) << 4) | ty.code());
        } else {
            self.buf.push(ty.code());
            self.varint(zigzag(i64::from(id)));
        }
        self.last = id;
        Ok(())
    }

    pub fn field_bool(&mut self, id: i16, value: bool) -> Result<()> {
        self.field(
            id,
            if value {
                Type::BoolTrue
            } else {
                Type::BoolFalse
            },
        )
    }

    pub fn field_i8(&mut self, id: i16, value: i8) -> Result<()> {
        self.field(id, Type::I8)?;
        self.i8(value);
        Ok(())
    }

    pub fn field_i16(&mut self, id: i16, value: i16) -> Result<()> {
        self.field(id, Type::I16)?;
        self.i16(value);
        Ok(())
    }

    pub fn field_i32(&mut self, id: i16, value: i32) -> Result<()> {
        self.field(id, Type::I32)?;
        self.i32(value);
        Ok(())
    }

    pub fn field_i64(&mut self, id: i16, value: i64) -> Result<()> {
        self.field(id, Type::I64)?;
        self.i64(value);
        Ok(())
    }

    pub fn field_double(&mut self, id: i16, value: f64) -> Result<()> {
        self.field(id, Type::Double)?;
        self.double(value);
        Ok(())
    }

    pub fn field_string(&mut self, id: i16, value: &str) -> Result<()> {
        self.field(id, Type::Binary)?;
        self.string(value);
        Ok(())
    }

    pub fn i8(&mut self, value: i8) {
        self.buf.push(value as u8);
    }

    pub fn i16(&mut self, value: i16) {
        self.varint(zigzag(i64::from(value)));
    }

    pub fn i32(&mut self, value: i32) {
        self.varint(zigzag(i64::from(value)));
    }

    pub fn i64(&mut self, value: i64) {
        self.varint(zigzag(value));
    }

    pub fn double(&mut self, value: f64) {
        self.buf.extend_from_slice(&value.to_le_bytes());
    }

    pub fn binary(&mut self, value: &[u8]) {
        self.varint(value.len() as u64);
        self.buf.extend_from_slice(value);
    }

    pub fn string(&mut self, value: &str) {
        self.binary(value.as_bytes());
    }

    /// A boolean list or map element (one byte).
    pub fn bool_element(&mut self, value: bool) {
        self.buf.push(if value { 1 } else { 2 });
    }

    /// List header; `elem` of a bool list is `Type::BoolTrue`.
    pub fn list_header(&mut self, elem: Type, size: usize) {
        if size < 15 {
            self.buf.push(((size as u8) << 4) | elem.code());
        } else {
            self.buf.push(0xf0 | elem.code());
            self.varint(size as u64);
        }
    }

    pub fn map_header(&mut self, key: Type, value: Type, size: usize) {
        self.varint(size as u64);
        if size > 0 {
            self.buf.push((key.code() << 4) | value.code());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip_i64(v: i64) {
        let mut w = Writer::new();
        w.i64(v);
        let bytes = w.finish().unwrap();
        let mut r = Reader::new(&bytes);
        assert_eq!(r.i64(Type::I64).unwrap(), v);
        r.finish().unwrap();
    }

    #[test]
    fn integers_round_trip_at_the_extremes() {
        for v in [0, 1, -1, 63, 64, -64, -65, 1 << 31, i64::MAX, i64::MIN] {
            roundtrip_i64(v);
        }
        for v in [0i32, 1, -1, i32::MAX, i32::MIN] {
            let mut w = Writer::new();
            w.i32(v);
            let b = w.finish().unwrap();
            assert_eq!(Reader::new(&b).i32(Type::I32).unwrap(), v);
        }
        for v in [0i16, 1, -1, i16::MAX, i16::MIN] {
            let mut w = Writer::new();
            w.i16(v);
            let b = w.finish().unwrap();
            assert_eq!(Reader::new(&b).i16(Type::I16).unwrap(), v);
        }
    }

    #[test]
    fn known_encodings() {
        let mut w = Writer::new();
        w.i32(1);
        w.i32(-1);
        w.i32(300);
        assert_eq!(w.finish().unwrap(), vec![2, 1, 0xd8, 0x04]);
    }

    #[test]
    fn out_of_range_narrow_integers_are_errors() {
        let mut w = Writer::new();
        w.i64(i64::from(i32::MAX) + 1);
        let b = w.finish().unwrap();
        assert!(Reader::new(&b).i32(Type::I32).is_err());
        let mut w = Writer::new();
        w.i32(i32::from(i16::MAX) + 1);
        let b = w.finish().unwrap();
        assert!(Reader::new(&b).i16(Type::I16).is_err());
    }

    #[test]
    fn varint_overflow_and_length_are_errors() {
        let eleven = [0x80u8; 10]
            .iter()
            .chain([&1u8])
            .copied()
            .collect::<Vec<_>>();
        assert!(Reader::new(&eleven).i64(Type::I64).is_err());
        // Ten bytes whose last byte carries more than one bit overflow 64 bits.
        let mut over = vec![0xffu8; 9];
        over.push(0x02);
        assert!(Reader::new(&over).i64(Type::I64).is_err());
        let mut max = vec![0xffu8; 9];
        max.push(0x01);
        assert!(Reader::new(&max).i64(Type::I64).is_ok());
    }

    #[test]
    fn doubles_are_little_endian_and_bit_exact() {
        for v in [
            0.0,
            -0.0,
            1.5,
            f64::NAN,
            f64::INFINITY,
            f64::MIN_POSITIVE / 2.0,
        ] {
            let mut w = Writer::new();
            w.double(v);
            let b = w.finish().unwrap();
            assert_eq!(b, v.to_le_bytes());
            let got = Reader::new(&b).double(Type::Double).unwrap();
            assert_eq!(got.to_bits(), v.to_bits());
        }
    }

    #[test]
    fn field_headers_use_short_and_long_forms() {
        let mut w = Writer::new();
        w.struct_begin();
        w.field_i32(1, 7).unwrap(); // delta 1: short
        w.field_i32(16, 8).unwrap(); // delta 15: short
        w.field_i32(32, 9).unwrap(); // delta 16: long
        w.field_bool(33, true).unwrap();
        w.field_bool(40, false).unwrap();
        w.field_i32(20000, 1).unwrap();
        w.struct_end().unwrap();
        let b = w.finish().unwrap();
        assert_eq!(b[0], 0x15);
        assert_eq!(b[2], 0xf5);
        assert_eq!(b[4], 5); // long form: type nibble only, then zigzag id
        let mut r = Reader::new(&b);
        let mut seen = Vec::new();
        r.read_struct(|r, id, ty| {
            match ty {
                Type::I32 => seen.push((id, i64::from(r.i32(ty)?))),
                Type::BoolTrue | Type::BoolFalse => seen.push((id, i64::from(r.boolean(ty)?))),
                _ => panic!("unexpected {ty:?}"),
            }
            Ok(true)
        })
        .unwrap();
        r.finish().unwrap();
        assert_eq!(
            seen,
            vec![(1, 7), (16, 8), (32, 9), (33, 1), (40, 0), (20000, 1)]
        );
    }

    #[test]
    fn writer_rejects_descending_and_repeated_ids() {
        let mut w = Writer::new();
        w.struct_begin();
        w.field_i32(3, 0).unwrap();
        assert!(w.field_i32(3, 0).is_err());
        assert!(w.field_i32(2, 0).is_err());
        let mut w = Writer::new();
        assert!(w.struct_end().is_err());
        let mut w = Writer::new();
        w.struct_begin();
        assert!(w.finish().is_err());
    }

    #[test]
    fn nested_structs_restore_the_field_id() {
        let mut w = Writer::new();
        w.struct_begin();
        w.field_i32(1, 1).unwrap();
        w.field(2, Type::Struct).unwrap();
        w.struct_begin();
        w.field_i32(1, 5).unwrap();
        w.struct_end().unwrap();
        w.field_i32(3, 3).unwrap();
        w.struct_end().unwrap();
        let b = w.finish().unwrap();
        let mut r = Reader::new(&b);
        let mut ids = Vec::new();
        r.read_struct(|r, id, ty| {
            ids.push(id);
            if ty == Type::Struct {
                r.read_struct(|r, id, ty| {
                    ids.push(100 + id);
                    r.i32(ty)?;
                    Ok(true)
                })?;
            } else {
                r.i32(ty)?;
            }
            Ok(true)
        })
        .unwrap();
        assert_eq!(ids, vec![1, 2, 101, 3]);
    }

    #[test]
    fn lists_use_short_and_long_headers() {
        for n in [0usize, 1, 14, 15, 16, 300] {
            let mut w = Writer::new();
            w.list_header(Type::I32, n);
            for i in 0..n {
                w.i32(i as i32 - 7);
            }
            let b = w.finish().unwrap();
            if n < 15 {
                assert_eq!(b[0], ((n as u8) << 4) | 5);
            } else {
                assert_eq!(b[0], 0xf5);
            }
            let mut r = Reader::new(&b);
            let got = r.list_i32(Type::List).unwrap();
            r.finish().unwrap();
            assert_eq!(got.len(), n);
            assert!(got.iter().enumerate().all(|(i, &v)| v == i as i32 - 7));
        }
    }

    #[test]
    fn string_lists_and_binary_round_trip() {
        let mut w = Writer::new();
        w.list_header(Type::Binary, 2);
        w.string("alpha");
        w.string("été");
        let b = w.finish().unwrap();
        let got = Reader::new(&b).list_string(Type::List).unwrap();
        assert_eq!(got, vec!["alpha".to_owned(), "été".to_owned()]);
        let mut w = Writer::new();
        w.binary(&[0xff, 0xfe]);
        let b = w.finish().unwrap();
        assert!(Reader::new(&b).string(Type::Binary).is_err());
        assert_eq!(Reader::new(&b).binary(Type::Binary).unwrap(), &[0xff, 0xfe]);
    }

    #[test]
    fn a_wrong_field_type_is_an_error() {
        let mut w = Writer::new();
        w.string("x");
        let b = w.finish().unwrap();
        assert!(Reader::new(&b).i32(Type::Binary).is_err());
        assert!(Reader::new(&b).list_i32(Type::Binary).is_err());
        assert!(Reader::new(&b).boolean(Type::I32).is_err());
    }

    /// A struct holding one field of every Thrift type, with nested containers.
    fn all_types() -> Vec<u8> {
        let mut w = Writer::new();
        w.struct_begin();
        w.field_bool(1, true).unwrap();
        w.field_bool(2, false).unwrap();
        w.field_i8(3, -5).unwrap();
        w.field_i16(4, -300).unwrap();
        w.field_i32(5, 70000).unwrap();
        w.field_i64(6, -(1 << 40)).unwrap();
        w.field_double(7, 2.5).unwrap();
        w.field_string(8, "hello").unwrap();
        w.field(9, Type::List).unwrap();
        w.list_header(Type::BoolTrue, 3);
        w.bool_element(true);
        w.bool_element(false);
        w.bool_element(true);
        w.field(10, Type::Set).unwrap();
        w.list_header(Type::I64, 2);
        w.i64(1);
        w.i64(-2);
        w.field(11, Type::Map).unwrap();
        w.map_header(Type::Binary, Type::List, 1);
        w.string("k");
        w.list_header(Type::Struct, 2);
        for _ in 0..2 {
            w.struct_begin();
            w.field_i32(1, 1).unwrap();
            w.field(2, Type::Struct).unwrap();
            w.struct_begin();
            w.field_bool(1, true).unwrap();
            w.struct_end().unwrap();
            w.struct_end().unwrap();
        }
        w.field(12, Type::Map).unwrap();
        w.map_header(Type::I32, Type::I32, 0);
        w.field(13, Type::Map).unwrap();
        w.map_header(Type::BoolTrue, Type::Double, 1);
        w.bool_element(true);
        w.double(1.0);
        w.field(14, Type::Struct).unwrap();
        w.struct_begin();
        w.struct_end().unwrap();
        w.field_i32(15, 9).unwrap();
        w.struct_end().unwrap();
        w.finish().unwrap()
    }

    #[test]
    fn skip_handles_every_thrift_type() {
        let b = all_types();
        let mut r = Reader::new(&b);
        r.skip(Type::Struct).unwrap();
        r.finish().unwrap();
        // The same struct read field by field sees the last field after skips.
        let mut r = Reader::new(&b);
        let mut last = None;
        r.read_struct(|r, id, ty| {
            if id == 15 {
                last = Some(r.i32(ty)?);
                Ok(true)
            } else {
                Ok(false)
            }
        })
        .unwrap();
        assert_eq!(last, Some(9));
    }

    #[test]
    fn every_truncation_of_a_skipped_struct_is_an_error() {
        let b = all_types();
        for len in 0..b.len() {
            let mut r = Reader::new(&b[..len]);
            assert!(r.skip(Type::Struct).is_err(), "prefix {len} parsed");
        }
    }

    #[test]
    fn nesting_beyond_the_limit_is_an_error() {
        let nest = |levels: usize| {
            let mut w = Writer::new();
            for _ in 0..levels {
                w.struct_begin();
                w.field(1, Type::Struct).unwrap();
            }
            // innermost struct is empty
            w.struct_begin();
            w.struct_end().unwrap();
            for _ in 0..levels {
                w.struct_end().unwrap();
            }
            w.finish().unwrap()
        };
        let ok = nest(MAX_THRIFT_DEPTH - 1);
        Reader::new(&ok).skip(Type::Struct).unwrap();
        let deep = nest(MAX_THRIFT_DEPTH);
        assert!(Reader::new(&deep).skip(Type::Struct).is_err());
        let hostile = nest(10_000);
        assert!(Reader::new(&hostile).skip(Type::Struct).is_err());
        // Lists of lists count too.
        let mut b = vec![0x19u8; 1000]; // each byte: a list of one list
        b.push(0x05);
        b.push(0x00);
        assert!(Reader::new(&b).skip(Type::List).is_err());
    }

    #[test]
    fn lying_lengths_are_errors_not_allocations() {
        // List claiming 2^60 elements.
        let mut b = vec![0xf5];
        b.extend_from_slice(&[0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x0f]);
        assert!(Reader::new(&b).list_i32(Type::List).is_err());
        assert!(Reader::new(&b).skip(Type::List).is_err());
        // Binary claiming more bytes than exist.
        let b = [0xe8, 0x07, 1, 2];
        assert!(Reader::new(&b).binary(Type::Binary).is_err());
        // Map with a huge count.
        let b = [0xff, 0xff, 0xff, 0xff, 0x0f, 0x55];
        assert!(Reader::new(&b).skip(Type::Map).is_err());
        // List element type 0 (stop) and invalid type codes.
        assert!(Reader::new(&[0x10]).list_header(Type::List).is_err());
        assert!(Reader::new(&[0x20, 0, 0]).list_header(Type::List).is_err());
        // An empty list written with element type 0 is accepted and empty.
        let mut r = Reader::new(&[0x00]);
        assert_eq!(r.list_header(Type::List).unwrap(), (Type::Stop, 0));
        assert!(Reader::new(&[0x00])
            .list_i32(Type::List)
            .unwrap()
            .is_empty());
        Reader::new(&[0x00]).skip(Type::List).unwrap();
        assert!(Reader::new(&[0x1d, 0]).list_header(Type::List).is_err());
        // Field header with delta but a stop type.
        let mut last = 0;
        assert!(Reader::new(&[0x10]).field_header(&mut last).is_err());
    }

    #[test]
    fn the_memory_bound_stops_a_hostile_expansion() {
        // 5000 ordinary short strings stay far inside the bound.
        let mut w = Writer::new();
        w.list_header(Type::Binary, 5000);
        for _ in 0..5000 {
            w.string("abcdefgh");
        }
        let b = w.finish().unwrap();
        let mut r = Reader::new(&b);
        assert!(r.list_string(Type::List).is_ok());
        assert!(r.charged() <= parse_memory_bound(b.len()));
        // 5000 empty strings cost 24 bytes each for one input byte each: refused.
        let mut w = Writer::new();
        w.list_header(Type::Binary, 5000);
        for _ in 0..5000 {
            w.string("");
        }
        let b = w.finish().unwrap();
        assert!(Reader::new(&b).list_string(Type::List).is_err());
        assert!(Reader::with_limit(&b, 1000).skip(Type::List).is_ok());
    }

    #[test]
    fn trailing_bytes_are_reported() {
        let mut r = Reader::new(&[1, 2]);
        r.skip(Type::I8).unwrap();
        assert!(r.finish().is_err());
    }
}
