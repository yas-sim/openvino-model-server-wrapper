# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: tensorflow/core/protobuf/saver.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='tensorflow/core/protobuf/saver.proto',
  package='tensorflow',
  syntax='proto3',
  serialized_options=b'\n\023org.tensorflow.utilB\013SaverProtosP\001ZUgithub.com/tensorflow/tensorflow/tensorflow/go/core/protobuf/for_core_protos_go_proto\370\001\001',
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n$tensorflow/core/protobuf/saver.proto\x12\ntensorflow\"\x9e\x02\n\x08SaverDef\x12\x1c\n\x14\x66ilename_tensor_name\x18\x01 \x01(\t\x12\x18\n\x10save_tensor_name\x18\x02 \x01(\t\x12\x17\n\x0frestore_op_name\x18\x03 \x01(\t\x12\x13\n\x0bmax_to_keep\x18\x04 \x01(\x05\x12\x0f\n\x07sharded\x18\x05 \x01(\x08\x12%\n\x1dkeep_checkpoint_every_n_hours\x18\x06 \x01(\x02\x12=\n\x07version\x18\x07 \x01(\x0e\x32,.tensorflow.SaverDef.CheckpointFormatVersion\"5\n\x17\x43heckpointFormatVersion\x12\n\n\x06LEGACY\x10\x00\x12\x06\n\x02V1\x10\x01\x12\x06\n\x02V2\x10\x02\x42~\n\x13org.tensorflow.utilB\x0bSaverProtosP\x01ZUgithub.com/tensorflow/tensorflow/tensorflow/go/core/protobuf/for_core_protos_go_proto\xf8\x01\x01\x62\x06proto3'
)



_SAVERDEF_CHECKPOINTFORMATVERSION = _descriptor.EnumDescriptor(
  name='CheckpointFormatVersion',
  full_name='tensorflow.SaverDef.CheckpointFormatVersion',
  filename=None,
  file=DESCRIPTOR,
  create_key=_descriptor._internal_create_key,
  values=[
    _descriptor.EnumValueDescriptor(
      name='LEGACY', index=0, number=0,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='V1', index=1, number=1,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='V2', index=2, number=2,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=286,
  serialized_end=339,
)
_sym_db.RegisterEnumDescriptor(_SAVERDEF_CHECKPOINTFORMATVERSION)


_SAVERDEF = _descriptor.Descriptor(
  name='SaverDef',
  full_name='tensorflow.SaverDef',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='filename_tensor_name', full_name='tensorflow.SaverDef.filename_tensor_name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='save_tensor_name', full_name='tensorflow.SaverDef.save_tensor_name', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='restore_op_name', full_name='tensorflow.SaverDef.restore_op_name', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='max_to_keep', full_name='tensorflow.SaverDef.max_to_keep', index=3,
      number=4, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='sharded', full_name='tensorflow.SaverDef.sharded', index=4,
      number=5, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='keep_checkpoint_every_n_hours', full_name='tensorflow.SaverDef.keep_checkpoint_every_n_hours', index=5,
      number=6, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='version', full_name='tensorflow.SaverDef.version', index=6,
      number=7, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _SAVERDEF_CHECKPOINTFORMATVERSION,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=53,
  serialized_end=339,
)

_SAVERDEF.fields_by_name['version'].enum_type = _SAVERDEF_CHECKPOINTFORMATVERSION
_SAVERDEF_CHECKPOINTFORMATVERSION.containing_type = _SAVERDEF
DESCRIPTOR.message_types_by_name['SaverDef'] = _SAVERDEF
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

SaverDef = _reflection.GeneratedProtocolMessageType('SaverDef', (_message.Message,), {
  'DESCRIPTOR' : _SAVERDEF,
  '__module__' : 'tensorflow.core.protobuf.saver_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.SaverDef)
  })
_sym_db.RegisterMessage(SaverDef)


DESCRIPTOR._options = None
# @@protoc_insertion_point(module_scope)
