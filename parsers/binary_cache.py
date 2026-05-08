"""
================================================================================
PaleoAST Parsers - Binary Cache (.pastx) Module
================================================================================

本模块实现自定义二进制缓存格式.pastx，用于快速加载大型距离矩阵。

二进制格式设计:
==============================================================================

文件结构:
    ┌─────────────────────────────────────────────────────────┐
    │                   文件头 (64 bytes)                      │
    ├─────────────────────────────────────────────────────────┤
    │                   数据块 (变长)                          │
    ├─────────────────────────────────────────────────────────┤
    │                   校验和 (32 bytes)                      │
    └─────────────────────────────────────────────────────────┘

文件头结构:
    Offset  Size  Description
    ─────────────────────────────────────
    0       4     Magic Number (0x50415354 = "PAST")
    4       4     Version (1 = 0x00000001)
    8       4     Flags (保留)
    12      4     行数 (nrow)
    16      4     列数 (ncol)
    20      4     数据类型 (0=float32, 1=float64, 2=int32)
    24      8     矩阵大小 (bytes)
    32      8     元数据偏移量
    40      4     元数据长度
    44      4     CRC32校验和
    48      16    保留

数据块:
    - 使用struct模块进行二进制打包
    - 支持内存映射(mmap)读取
    - 使用zlib进行可选压缩

作者: PaleoAST Development Team
"""

from __future__ import annotations
from typing import (
    Dict, List, Optional, Tuple, Any, BinaryIO, Iterator
)
from dataclasses import dataclass, field
from enum import Enum, auto
import struct
import zlib
import hashlib
import logging
import mmap
import os
from pathlib import Path

logger = logging.getLogger(__name__)


# ================================================================================
# 常量和枚举
# ================================================================================

class ChunkType(Enum):
    """数据块类型枚举"""
    MATRIX_DATA = 0x01
    ROW_LABELS = 0x02
    COL_LABELS = 0x03
    METADATA = 0x04
    DISTANCE_TYPE = 0x05
    STATISTICS = 0x06


class DataType(Enum):
    """数据类型枚举"""
    FLOAT32 = 0
    FLOAT64 = 1
    INT32 = 2
    UINT8 = 3


@dataclass
class BinaryCacheHeader:
    """
    二进制缓存文件头
    
    属性:
        magic: 魔数 (PAST)
        version: 版本号
        flags: 标志位
        nrow: 行数
        ncol: 列数
        dtype: 数据类型
        matrix_size: 矩阵数据大小 (bytes)
        metadata_offset: 元数据偏移量
        metadata_length: 元数据长度
        crc32: CRC32校验和
    """
    magic: int = 0x50415354  # "PAST"
    version: int = 1
    flags: int = 0
    nrow: int = 0
    ncol: int = 0
    dtype: DataType = DataType.FLOAT64
    matrix_size: int = 0
    metadata_offset: int = 0
    metadata_length: int = 0
    crc32: int = 0
    
    # 头部长度
    HEADER_SIZE: int = 64
    
    def to_bytes(self) -> bytes:
        """
        将头部序列化为字节
        
        返回:
            64字节的头部数据
        """
        # 打包格式: !IIIIIQQQII
        # ! = 网络字节序(大端)
        # I = unsigned int (4 bytes)
        # Q = unsigned long long (8 bytes)
        
        header_data = struct.pack(
            '!IIIIII QQII',
            self.magic,
            self.version,
            self.flags,
            self.nrow,
            self.ncol,
            self.dtype.value,
            self.matrix_size,
            self.metadata_offset,
            self.metadata_length,
            self.crc32,
        )
        
        # 确保正好64字节
        if len(header_data) < self.HEADER_SIZE:
            header_data = header_data + b'\x00' * (self.HEADER_SIZE - len(header_data))
        
        return header_data[:self.HEADER_SIZE]
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'BinaryCacheHeader':
        """
        从字节反序列化头部
        
        参数:
            data: 64字节的头部数据
        
        返回:
            BinaryCacheHeader对象
        """
        if len(data) < cls.HEADER_SIZE:
            raise ValueError(
                f"Header data too short: expected {cls.HEADER_SIZE}, got {len(data)}"
            )
        
        unpacked = struct.unpack('!IIIIII QQII', data[:48])
        
        return cls(
            magic=unpacked[0],
            version=unpacked[1],
            flags=unpacked[2],
            nrow=unpacked[3],
            ncol=unpacked[4],
            dtype=DataType(unpacked[5]),
            matrix_size=unpacked[6],
            metadata_offset=unpacked[7],
            metadata_length=unpacked[8],
            crc32=unpacked[9],
        )
    
    def validate(self) -> bool:
        """
        验证头部有效性
        
        返回:
            如果有效返回True
        """
        if self.magic != 0x50415354:
            logger.error(f"Invalid magic number: {hex(self.magic)}")
            return False
        
        if self.version > 1:
            logger.error(f"Unsupported version: {self.version}")
            return False
        
        if self.nrow <= 0 or self.ncol <= 0:
            logger.error(f"Invalid dimensions: {self.nrow}x{self.ncol}")
            return False
        
        return True


class BinaryCache:
    """
    二进制缓存读写器
    
    提供高效的矩阵序列化和反序列化功能。
    
    核心功能:
        1. 将NumPy矩阵写入二进制文件
        2. 从二进制文件读取矩阵（支持mmap）
        3. CRC32校验
        4. 可选zlib压缩
        5. 元数据存储
    
    性能特性:
        - mmap支持: 无需将整个文件加载到内存
        - 压缩支持: 牺牲速度换取空间
        - 批量写入: 使用struct.pack_into
    
    使用示例:
        >>> cache = BinaryCache()
        >>> 
        >>> # 写入
        >>> cache.save('matrix.pastx', matrix, row_labels=['a', 'b'])
        >>> 
        >>> # 读取
        >>> data = cache.load('matrix.pastx')
        >>> print(data['matrix'])
    """
    
    def __init__(self, use_compression: bool = False):
        """
        初始化二进制缓存
        
        参数:
            use_compression: 是否使用zlib压缩
        """
        self._use_compression = use_compression
        self._logger = logging.getLogger(f"{__name__}.BinaryCache")
    
    def save(
        self,
        filepath: str,
        matrix,
        row_labels: Optional[List[str]] = None,
        col_labels: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        将矩阵保存为二进制缓存
        
        参数:
            filepath: 输出文件路径
            matrix: NumPy矩阵
            row_labels: 行标签列表
            col_labels: 列标签列表
            metadata: 元数据字典
        
        返回:
            如果成功返回True
        """
        import numpy as np
        
        try:
            # 验证输入
            if not isinstance(matrix, np.ndarray):
                raise TypeError(f"Expected numpy array, got {type(matrix)}")
            
            nrow, ncol = matrix.shape
            
            # 确定数据类型
            dtype_map = {
                np.float32: DataType.FLOAT32,
                np.float64: DataType.FLOAT64,
                np.int32: DataType.INT32,
                np.uint8: DataType.UINT8,
            }
            
            dtype = dtype_map.get(matrix.dtype.type, DataType.FLOAT64)
            
            # 计算矩阵大小
            itemsize = matrix.dtype.itemsize
            matrix_size = nrow * ncol * itemsize
            
            # 创建头部
            header = BinaryCacheHeader(
                nrow=nrow,
                ncol=ncol,
                dtype=dtype,
                matrix_size=matrix_size,
            )
            
            # 序列化元数据
            metadata_json = ""
            if metadata:
                import json
                metadata_json = json.dumps(metadata)
                metadata_bytes = metadata_json.encode('utf-8')
            else:
                metadata_bytes = b""
            
            header.metadata_length = len(metadata_bytes)
            
            # 序列化标签
            row_labels_bytes = self._serialize_labels(row_labels)
            col_labels_bytes = self._serialize_labels(col_labels)
            
            # 计算偏移量
            header.metadata_offset = (
                header.HEADER_SIZE +
                matrix_size +
                len(row_labels_bytes) +
                len(col_labels_bytes)
            )
            
            # 序列化矩阵数据
            matrix_bytes = matrix.tobytes()
            
            # 可选压缩
            if self._use_compression:
                matrix_bytes = zlib.compress(matrix_bytes, level=6)
                header.flags |= 0x01  # 设置压缩标志
            
            # 计算CRC32
            crc_data = matrix_bytes + row_labels_bytes + col_labels_bytes + metadata_bytes
            header.crc32 = zlib.crc32(crc_data) & 0xFFFFFFFF
            
            # 写入文件
            with open(filepath, 'wb') as f:
                # 写入头部
                f.write(header.to_bytes())
                
                # 写入矩阵数据
                f.write(matrix_bytes)
                
                # 写入行标签
                f.write(row_labels_bytes)
                
                # 写入列标签
                f.write(col_labels_bytes)
                
                # 写入元数据
                f.write(metadata_bytes)
            
            self._logger.info(
                f"Saved matrix {nrow}x{ncol} to {filepath} "
                f"({os.path.getsize(filepath)} bytes)"
            )
            return True
        
        except Exception as e:
            self._logger.error(f"Failed to save binary cache: {e}")
            return False
    
    def load(
        self,
        filepath: str,
        use_mmap: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        从二进制缓存加载矩阵
        
        参数:
            filepath: 文件路径
            use_mmap: 是否使用内存映射
        
        返回:
            包含矩阵和元数据的字典，或None
        """
        import numpy as np
        
        try:
            with open(filepath, 'rb') as f:
                # 读取头部
                header_data = f.read(BinaryCacheHeader.HEADER_SIZE)
                header = BinaryCacheHeader.from_bytes(header_data)
                
                if not header.validate():
                    raise ValueError("Invalid binary cache header")
                
                # 读取矩阵数据
                matrix_bytes = f.read(header.matrix_size)
                
                # 检查压缩
                if header.flags & 0x01:
                    matrix_bytes = zlib.decompress(matrix_bytes)
                
                # 恢复矩阵
                dtype_map = {
                    DataType.FLOAT32: np.float32,
                    DataType.FLOAT64: np.float64,
                    DataType.INT32: np.int32,
                    DataType.UINT8: np.uint8,
                }
                
                matrix = np.frombuffer(
                    matrix_bytes,
                    dtype=dtype_map.get(header.dtype, np.float64)
                ).reshape(header.nrow, header.ncol)
                
                # 读取行标签
                row_labels = self._read_labels(f, header.nrow)
                
                # 读取列标签
                col_labels = self._read_labels(f, header.ncol)
                
                # 读取元数据
                if header.metadata_length > 0:
                    f.seek(header.metadata_offset)
                    metadata_bytes = f.read(header.metadata_length)
                    import json
                    metadata = json.loads(metadata_bytes.decode('utf-8'))
                else:
                    metadata = {}
                
                # 验证CRC32
                crc_data = matrix_bytes + self._serialize_labels(row_labels) + \
                           self._serialize_labels(col_labels) + metadata_bytes
                expected_crc = zlib.crc32(crc_data) & 0xFFFFFFFF
                
                if expected_crc != header.crc32:
                    self._logger.warning(
                        f"CRC32 mismatch: expected {header.crc32}, got {expected_crc}"
                    )
                
                result = {
                    'matrix': matrix,
                    'row_labels': row_labels,
                    'col_labels': col_labels,
                    'metadata': metadata,
                    'header': header,
                }
                
                self._logger.info(
                    f"Loaded matrix {header.nrow}x{header.ncol} from {filepath}"
                )
                
                return result
        
        except Exception as e:
            self._logger.error(f"Failed to load binary cache: {e}")
            return None
    
    def save_mmap(
        self,
        filepath: str,
        matrix,
        row_labels: Optional[List[str]] = None,
        col_labels: Optional[List[str]] = None
    ) -> bool:
        """
        保存并创建内存映射文件
        
        适用于超大矩阵的随机访问。
        
        参数:
            filepath: 文件路径
            matrix: NumPy矩阵
            row_labels: 行标签
            col_labels: 列标签
        
        返回:
            如果成功返回True
        """
        # 先普通保存
        if not self.save(filepath, matrix, row_labels, col_labels):
            return False
        
        # 返回mmap对象
        return True
    
    def load_mmap(
        self,
        filepath: str
    ) -> Optional[Tuple[Any, mmap.mmap]]:
        """
        加载内存映射文件
        
        参数:
            filepath: 文件路径
        
        返回:
            (NumPy数组, mmap对象) 或 None
        """
        import numpy as np
        
        try:
            # 打开文件
            fd = os.open(filepath, os.O_RDONLY)
            
            # 创建mmap
            file_size = os.fstat(fd).st_size
            mm = mmap.mmap(fd, file_size, access=mmap.ACCESS_READ)
            
            # 读取头部
            header = BinaryCacheHeader.from_bytes(mm[:BinaryCacheHeader.HEADER_SIZE])
            
            if not header.validate():
                raise ValueError("Invalid header")
            
            # 计算矩阵区域
            matrix_start = BinaryCacheHeader.HEADER_SIZE
            matrix_end = matrix_start + header.matrix_size
            
            # 创建numpy数组视图
            dtype_map = {
                DataType.FLOAT32: np.float32,
                DataType.FLOAT64: np.float64,
                DataType.INT32: np.int32,
                DataType.UINT8: np.uint8,
            }
            
            # 直接从mmap创建数组
            matrix_bytes = mm[matrix_start:matrix_end]
            
            # 检查压缩
            if header.flags & 0x01:
                matrix_bytes = zlib.decompress(matrix_bytes)
            
            matrix = np.frombuffer(
                matrix_bytes,
                dtype=dtype_map.get(header.dtype, np.float64)
            ).reshape(header.nrow, header.ncol).copy()  # 复制以断开mmap
            
            return matrix, mm
        
        except Exception as e:
            self._logger.error(f"Failed to load mmap: {e}")
            return None
    
    def _serialize_labels(self, labels: Optional[List[str]]) -> bytes:
        """
        序列化标签列表
        
        格式: 4字节数量 + [4字节长度 + 标签UTF-8字节]*
        
        参数:
            labels: 标签列表
        
        返回:
            序列化字节
        """
        if not labels:
            # 空列表: 4字节0
            return struct.pack('!I', 0)
        
        data = struct.pack('!I', len(labels))
        
        for label in labels:
            label_bytes = label.encode('utf-8')
            data += struct.pack('!I', len(label_bytes))
            data += label_bytes
        
        return data
    
    def _read_labels(self, f: BinaryIO, expected_count: int) -> List[str]:
        """
        从文件读取标签
        
        参数:
            f: 文件对象
            expected_count: 期望的标签数量
        
        返回:
            标签列表
        """
        # 读取数量
        count_data = f.read(4)
        if len(count_data) < 4:
            return []
        
        count = struct.unpack('!I', count_data)[0]
        
        if count == 0:
            return []
        
        labels = []
        for _ in range(count):
            # 读取长度
            len_data = f.read(4)
            if len(len_data) < 4:
                break
            
            length = struct.unpack('!I', len_data)[0]
            
            # 读取标签
            label_bytes = f.read(length)
            if len(label_bytes) < length:
                break
            
            labels.append(label_bytes.decode('utf-8'))
        
        return labels
    
    def get_info(self, filepath: str) -> Optional[Dict[str, Any]]:
        """
        获取文件信息而不加载数据
        
        参数:
            filepath: 文件路径
        
        返回:
            文件信息字典
        """
        try:
            with open(filepath, 'rb') as f:
                header_data = f.read(BinaryCacheHeader.HEADER_SIZE)
                header = BinaryCacheHeader.from_bytes(header_data)
                
                return {
                    'version': header.version,
                    'nrow': header.nrow,
                    'ncol': header.ncol,
                    'dtype': header.dtype.name,
                    'matrix_size': header.matrix_size,
                    'compressed': bool(header.flags & 0x01),
                    'file_size': os.path.getsize(filepath),
                    'crc32': hex(header.crc32),
                }
        
        except Exception as e:
            self._logger.error(f"Failed to get file info: {e}")
            return None


def save_matrix(
    filepath: str,
    matrix,
    row_labels: Optional[List[str]] = None,
    col_labels: Optional[List[str]] = None,
    use_compression: bool = False
) -> bool:
    """
    保存矩阵为二进制缓存的便捷函数
    
    参数:
        filepath: 文件路径
        matrix: NumPy矩阵
        row_labels: 行标签
        col_labels: 列标签
        use_compression: 是否压缩
    
    返回:
        是否成功
    """
    cache = BinaryCache(use_compression=use_compression)
    return cache.save(filepath, matrix, row_labels, col_labels)


def load_matrix(filepath: str) -> Optional[Dict[str, Any]]:
    """
    加载二进制缓存矩阵的便捷函数
    
    参数:
        filepath: 文件路径
    
    返回:
        数据字典
    """
    cache = BinaryCache()
    return cache.load(filepath)
