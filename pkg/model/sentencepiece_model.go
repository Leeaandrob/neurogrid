// Package model provides model loading and weight management for LLM inference.
package model

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
)

// SentencePieceModel represents a parsed SentencePiece model.
type SentencePieceModel struct {
	Pieces         []SentencePiecePiece
	TrainerSpec    *TrainerSpec
	NormalizerSpec *NormalizerSpec
}

// SentencePiecePiece represents a vocabulary piece in the model.
type SentencePiecePiece struct {
	Piece string
	Score float32
	Type  PieceType
}

// PieceType represents the type of a vocabulary piece.
type PieceType int

const (
	PieceTypeNormal      PieceType = 1
	PieceTypeUnknown     PieceType = 2
	PieceTypeControl     PieceType = 3
	PieceTypeUserDefined PieceType = 4
	PieceTypeByte        PieceType = 6
)

// TrainerSpec contains training configuration.
type TrainerSpec struct {
	VocabSize int32
	BosID     int32
	EosID     int32
	PadID     int32
	UnkID     int32
	BosString string
	EosString string
	PadString string
	UnkString string
}

// NormalizerSpec contains normalization configuration.
type NormalizerSpec struct {
	AddDummyPrefix         bool
	RemoveExtraWhitespaces bool
	EscapeWhitespaces      bool
}

// ParseSentencePieceModel parses a SentencePiece model file (protobuf format).
// This is a simplified parser that extracts vocabulary and special tokens.
func ParseSentencePieceModel(path string) (*SentencePieceModel, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read model file: %w", err)
	}

	return ParseSentencePieceModelData(data)
}

// ParseSentencePieceModelData parses SentencePiece model from raw bytes.
func ParseSentencePieceModelData(data []byte) (*SentencePieceModel, error) {
	if len(data) < 4 {
		return nil, errors.New("model data too short")
	}

	model := &SentencePieceModel{
		TrainerSpec: &TrainerSpec{
			BosID: 1,
			EosID: 2,
			PadID: -1,
			UnkID: 0,
		},
		NormalizerSpec: &NormalizerSpec{
			AddDummyPrefix:    true,
			EscapeWhitespaces: true,
		},
	}

	// Parse protobuf wire format
	reader := &protoReader{data: data}

	for !reader.EOF() {
		fieldNum, wireType, err := reader.ReadTag()
		if err != nil {
			if err == io.EOF {
				break
			}
			return nil, fmt.Errorf("failed to read tag: %w", err)
		}

		switch fieldNum {
		case 1: // pieces (repeated SentencePiece)
			if wireType != 2 { // length-delimited
				if err := reader.SkipField(wireType); err != nil {
					return nil, err
				}
				continue
			}
			piece, err := reader.ReadPiece()
			if err != nil {
				return nil, fmt.Errorf("failed to read piece: %w", err)
			}
			model.Pieces = append(model.Pieces, piece)

		case 2: // trainer_spec
			if wireType != 2 {
				if err := reader.SkipField(wireType); err != nil {
					return nil, err
				}
				continue
			}
			spec, err := reader.ReadTrainerSpec()
			if err != nil {
				return nil, fmt.Errorf("failed to read trainer_spec: %w", err)
			}
			model.TrainerSpec = spec

		case 3: // normalizer_spec
			if wireType != 2 {
				if err := reader.SkipField(wireType); err != nil {
					return nil, err
				}
				continue
			}
			// Skip normalizer spec for now
			length, err := reader.ReadVarint()
			if err != nil {
				return nil, err
			}
			reader.pos += int(length)

		default:
			if err := reader.SkipField(wireType); err != nil {
				return nil, err
			}
		}
	}

	return model, nil
}

// protoReader is a simple protobuf wire format reader.
type protoReader struct {
	data []byte
	pos  int
}

func (r *protoReader) EOF() bool {
	return r.pos >= len(r.data)
}

func (r *protoReader) ReadByte() (byte, error) {
	if r.pos >= len(r.data) {
		return 0, io.EOF
	}
	b := r.data[r.pos]
	r.pos++
	return b, nil
}

func (r *protoReader) ReadVarint() (uint64, error) {
	var result uint64
	var shift uint
	for {
		b, err := r.ReadByte()
		if err != nil {
			return 0, err
		}
		result |= uint64(b&0x7F) << shift
		if b < 0x80 {
			break
		}
		shift += 7
	}
	return result, nil
}

func (r *protoReader) ReadSignedVarint() (int64, error) {
	v, err := r.ReadVarint()
	if err != nil {
		return 0, err
	}
	// ZigZag decode
	return int64((v >> 1) ^ -(v & 1)), nil
}

func (r *protoReader) ReadTag() (fieldNum int, wireType int, err error) {
	tag, err := r.ReadVarint()
	if err != nil {
		return 0, 0, err
	}
	return int(tag >> 3), int(tag & 0x7), nil
}

func (r *protoReader) ReadBytes() ([]byte, error) {
	length, err := r.ReadVarint()
	if err != nil {
		return nil, err
	}
	if r.pos+int(length) > len(r.data) {
		return nil, io.ErrUnexpectedEOF
	}
	result := r.data[r.pos : r.pos+int(length)]
	r.pos += int(length)
	return result, nil
}

func (r *protoReader) ReadString() (string, error) {
	b, err := r.ReadBytes()
	if err != nil {
		return "", err
	}
	return string(b), nil
}

func (r *protoReader) ReadFixed32() (uint32, error) {
	if r.pos+4 > len(r.data) {
		return 0, io.ErrUnexpectedEOF
	}
	v := binary.LittleEndian.Uint32(r.data[r.pos:])
	r.pos += 4
	return v, nil
}

func (r *protoReader) ReadFloat32() (float32, error) {
	bits, err := r.ReadFixed32()
	if err != nil {
		return 0, err
	}
	return math.Float32frombits(bits), nil
}

func (r *protoReader) SkipField(wireType int) error {
	switch wireType {
	case 0: // varint
		_, err := r.ReadVarint()
		return err
	case 1: // 64-bit
		if r.pos+8 > len(r.data) {
			return io.ErrUnexpectedEOF
		}
		r.pos += 8
		return nil
	case 2: // length-delimited
		length, err := r.ReadVarint()
		if err != nil {
			return err
		}
		r.pos += int(length)
		return nil
	case 5: // 32-bit
		if r.pos+4 > len(r.data) {
			return io.ErrUnexpectedEOF
		}
		r.pos += 4
		return nil
	default:
		return fmt.Errorf("unknown wire type: %d", wireType)
	}
}

func (r *protoReader) ReadPiece() (SentencePiecePiece, error) {
	piece := SentencePiecePiece{
		Type: PieceTypeNormal,
	}

	length, err := r.ReadVarint()
	if err != nil {
		return piece, err
	}

	end := r.pos + int(length)
	if end > len(r.data) {
		return piece, io.ErrUnexpectedEOF
	}

	for r.pos < end {
		fieldNum, wireType, err := r.ReadTag()
		if err != nil {
			return piece, err
		}

		switch fieldNum {
		case 1: // piece (string)
			if wireType != 2 {
				if err := r.SkipField(wireType); err != nil {
					return piece, err
				}
				continue
			}
			piece.Piece, err = r.ReadString()
			if err != nil {
				return piece, err
			}

		case 2: // score (float)
			if wireType != 5 {
				if err := r.SkipField(wireType); err != nil {
					return piece, err
				}
				continue
			}
			piece.Score, err = r.ReadFloat32()
			if err != nil {
				return piece, err
			}

		case 3: // type (enum)
			if wireType != 0 {
				if err := r.SkipField(wireType); err != nil {
					return piece, err
				}
				continue
			}
			t, err := r.ReadVarint()
			if err != nil {
				return piece, err
			}
			piece.Type = PieceType(t)

		default:
			if err := r.SkipField(wireType); err != nil {
				return piece, err
			}
		}
	}

	return piece, nil
}

func (r *protoReader) ReadTrainerSpec() (*TrainerSpec, error) {
	spec := &TrainerSpec{
		BosID: 1,
		EosID: 2,
		PadID: -1,
		UnkID: 0,
	}

	length, err := r.ReadVarint()
	if err != nil {
		return spec, err
	}

	end := r.pos + int(length)
	if end > len(r.data) {
		return spec, io.ErrUnexpectedEOF
	}

	for r.pos < end {
		fieldNum, wireType, err := r.ReadTag()
		if err != nil {
			return spec, err
		}

		switch fieldNum {
		case 1: // vocab_size (int32)
			if wireType != 0 {
				if err := r.SkipField(wireType); err != nil {
					return spec, err
				}
				continue
			}
			v, err := r.ReadVarint()
			if err != nil {
				return spec, err
			}
			spec.VocabSize = int32(v)

		case 41: // bos_id (int32)
			if wireType != 0 {
				if err := r.SkipField(wireType); err != nil {
					return spec, err
				}
				continue
			}
			v, err := r.ReadVarint()
			if err != nil {
				return spec, err
			}
			spec.BosID = int32(v)

		case 42: // eos_id (int32)
			if wireType != 0 {
				if err := r.SkipField(wireType); err != nil {
					return spec, err
				}
				continue
			}
			v, err := r.ReadVarint()
			if err != nil {
				return spec, err
			}
			spec.EosID = int32(v)

		case 43: // pad_id (int32)
			if wireType != 0 {
				if err := r.SkipField(wireType); err != nil {
					return spec, err
				}
				continue
			}
			v, err := r.ReadSignedVarint()
			if err != nil {
				return spec, err
			}
			spec.PadID = int32(v)

		case 40: // unk_id (int32)
			if wireType != 0 {
				if err := r.SkipField(wireType); err != nil {
					return spec, err
				}
				continue
			}
			v, err := r.ReadVarint()
			if err != nil {
				return spec, err
			}
			spec.UnkID = int32(v)

		default:
			if err := r.SkipField(wireType); err != nil {
				return spec, err
			}
		}
	}

	return spec, nil
}
