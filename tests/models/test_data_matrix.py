"""
Tests for DataMatrix metadata functionality.

这些测试验证DataMatrix类的元数据功能：
1. metadata字段的创建和保留
2. specimen_metadata的创建和修改
3. column_metadata的创建和访问
4. to_dict/from_dict往返不丢失元数据
"""

from __future__ import annotations

import numpy as np
import pytest

from models.data_matrix import DataMatrix


class TestDataMatrixMetadata:
    """测试DataMatrix的元数据功能"""

    def test_metadata_init(self):
        """测试初始化时传入metadata"""
        meta = {"project": "Cambrian Explosion", "analyst": "Dr. Smith"}
        matrix = DataMatrix(
            [[1.0, 2.0], [3.0, 4.0]],
            row_labels=["A", "B"],
            col_labels=["X", "Y"],
            metadata=meta,
        )
        assert matrix.metadata == meta
        assert matrix.metadata["project"] == "Cambrian Explosion"
        # 验证原始字典修改不影响matrix
        meta["project"] = "Modified"
        assert matrix.metadata["project"] == "Cambrian Explosion"

    def test_specimen_metadata_init(self):
        """测试初始化时传入specimen_metadata"""
        spec_meta = [
            {"specimen_id": "AMNH-001", "formation": "Burgess Shale"},
            {"specimen_id": "USNM-002", "formation": "Wheeler Shale"},
        ]
        matrix = DataMatrix(
            [[1.0, 2.0], [3.0, 4.0]],
            row_labels=["A", "B"],
            col_labels=["X", "Y"],
            specimen_metadata=spec_meta,
        )
        assert len(matrix.specimen_metadata) == 2
        assert matrix.specimen_metadata[0]["specimen_id"] == "AMNH-001"
        assert matrix.specimen_metadata[1]["formation"] == "Wheeler Shale"

    def test_column_metadata_init(self):
        """测试初始化时传入column_metadata"""
        col_meta = {
            "X": {"description": "Carapace length", "units": "mm"},
            "Y": {"description": "Body width", "units": "mm"},
        }
        matrix = DataMatrix(
            [[1.0, 2.0], [3.0, 4.0]],
            row_labels=["A", "B"],
            col_labels=["X", "Y"],
            column_metadata=col_meta,
        )
        assert matrix.column_metadata["X"]["units"] == "mm"
        assert matrix.column_metadata["Y"]["description"] == "Body width"

    def test_metadata_defaults(self):
        """测试默认metadata为空字典"""
        matrix = DataMatrix(
            [[1.0, 2.0], [3.0, 4.0]],
            row_labels=["A", "B"],
        )
        assert matrix.metadata == {}
        assert matrix.specimen_metadata == [{}, {}]
        assert matrix.column_metadata == {"Var_1": {}, "Var_2": {}}

    def test_get_specimen_metadata_by_index(self):
        """测试通过索引获取specimen_metadata"""
        spec_meta = [{"id": "S1"}, {"id": "S2"}]
        matrix = DataMatrix(
            [[1.0], [2.0]],
            specimen_metadata=spec_meta,
        )
        result = matrix.get_specimen_metadata(0)
        assert result["id"] == "S1"
        result = matrix.get_specimen_metadata(1)
        assert result["id"] == "S2"

    def test_get_specimen_metadata_by_label(self):
        """测试通过行标签获取specimen_metadata"""
        spec_meta = [{"id": "S1"}, {"id": "S2"}]
        matrix = DataMatrix(
            [[1.0], [2.0]],
            row_labels=["A", "B"],
            specimen_metadata=spec_meta,
        )
        result = matrix.get_specimen_metadata("A")
        assert result["id"] == "S1"
        result = matrix.get_specimen_metadata("B")
        assert result["id"] == "S2"

    def test_get_specimen_metadata_invalid(self):
        """测试获取无效specimen_metadata"""
        matrix = DataMatrix([[1.0], [2.0]])
        with pytest.raises(IndexError):
            matrix.get_specimen_metadata(10)
        with pytest.raises(ValueError):
            matrix.get_specimen_metadata("NonExistent")

    def test_set_specimen_metadata(self):
        """测试设置specimen_metadata"""
        matrix = DataMatrix(
            [[1.0], [2.0]],
            row_labels=["A", "B"],
        )
        matrix.set_specimen_metadata(0, "collector", "Dr. Smith")
        matrix.set_specimen_metadata("B", "year", 2024)

        assert matrix.specimen_metadata[0]["collector"] == "Dr. Smith"
        assert matrix.specimen_metadata[1]["year"] == 2024

    def test_get_column_metadata_by_index(self):
        """测试通过索引获取column_metadata"""
        col_meta = {"X": {"units": "mm"}, "Y": {"units": "cm"}}
        matrix = DataMatrix(
            [[1.0, 2.0]],
            col_labels=["X", "Y"],
            column_metadata=col_meta,
        )
        result = matrix.get_column_metadata(0)
        assert result["units"] == "mm"
        result = matrix.get_column_metadata(1)
        assert result["units"] == "cm"

    def test_get_column_metadata_by_label(self):
        """测试通过列标签获取column_metadata"""
        col_meta = {"X": {"units": "mm"}, "Y": {"units": "cm"}}
        matrix = DataMatrix(
            [[1.0, 2.0]],
            col_labels=["X", "Y"],
            column_metadata=col_meta,
        )
        result = matrix.get_column_metadata("X")
        assert result["units"] == "mm"

    def test_to_dict_with_metadata(self):
        """测试to_dict包含metadata"""
        meta = {"project": "Test"}
        spec_meta = [{"id": "S1"}]
        col_meta = {"X": {"units": "mm"}}
        matrix = DataMatrix(
            [[1.0]],
            row_labels=["A"],
            col_labels=["X"],
            metadata=meta,
            specimen_metadata=spec_meta,
            column_metadata=col_meta,
        )
        d = matrix.to_dict()
        assert d["metadata"] == meta
        assert d["specimen_metadata"] == spec_meta
        assert d["column_metadata"] == col_meta

    def test_from_dict_with_metadata(self):
        """测试from_dict恢复metadata"""
        original = DataMatrix(
            [[1.0, 2.0], [3.0, 4.0]],
            row_labels=["A", "B"],
            col_labels=["X", "Y"],
            metadata={"project": "Test"},
            specimen_metadata=[{"id": "A"}, {"id": "B"}],
            column_metadata={"X": {"units": "mm"}, "Y": {"units": "cm"}},
        )
        d = original.to_dict()
        restored = DataMatrix.from_dict(d)

        assert restored.metadata == original.metadata
        assert restored.specimen_metadata == original.specimen_metadata
        assert restored.column_metadata == original.column_metadata
        assert restored.name == original.name
        assert restored.n_samples == original.n_samples
        assert restored.n_variables == original.n_variables

    def test_roundtrip_preserves_nested_metadata(self):
        """测试往返不丢失嵌套元数据"""
        nested_meta = {
            "project": "Test",
            "nested": {"level": 1, "data": [1, 2, 3]},
        }
        spec_meta = [
            {"id": "S1", "geo": {"lat": 40.0, "lon": -100.0}},
            {"id": "S2", "geo": {"lat": 41.0, "lon": -101.0}},
        ]
        col_meta = {
            "X": {"description": "Test", "coding": {"0": "absent", "1": "present"}}
        }

        original = DataMatrix(
            [[0, 1], [1, 0]],
            row_labels=["A", "B"],
            col_labels=["X", "Y"],
            metadata=nested_meta,
            specimen_metadata=spec_meta,
            column_metadata=col_meta,
        )

        d = original.to_dict()
        restored = DataMatrix.from_dict(d)

        assert restored.metadata == nested_meta
        assert restored.specimen_metadata[0]["geo"]["lat"] == 40.0
        assert restored.column_metadata["X"]["coding"]["0"] == "absent"

    def test_copy_preserves_metadata(self):
        """测试copy保留metadata"""
        matrix = DataMatrix(
            [[1.0, 2.0]],
            row_labels=["A"],
            col_labels=["X", "Y"],
            metadata={"project": "Test"},
            specimen_metadata=[{"id": "S1"}],
            column_metadata={"X": {"units": "mm"}},
        )
        copy = matrix.copy()
        assert copy.metadata == matrix.metadata
        assert copy.specimen_metadata == matrix.specimen_metadata
        assert copy.column_metadata == matrix.column_metadata

    def test_subset_rows_preserves_metadata(self):
        """测试subset_rows保留specimen_metadata"""
        matrix = DataMatrix(
            [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],  # 3 rows, 2 columns
            row_labels=["A", "B", "C"],
            col_labels=["X", "Y"],  # Match actual columns
            specimen_metadata=[{"id": "S1"}, {"id": "S2"}, {"id": "S3"}],
            column_metadata={"X": {"units": "mm"}, "Y": {"units": "cm"}},
        )
        subset = matrix.subset_rows([0, 2])
        assert subset.n_samples == 2
        assert subset.specimen_metadata[0]["id"] == "S1"
        assert subset.specimen_metadata[1]["id"] == "S3"
        # column_metadata should be preserved
        assert subset.column_metadata["X"]["units"] == "mm"
        assert subset.column_metadata["Y"]["units"] == "cm"

    def test_subset_columns_preserves_column_metadata(self):
        """测试subset_columns保留column_metadata"""
        matrix = DataMatrix(
            [[1.0, 2.0, 3.0]],
            col_labels=["X", "Y", "Z"],
            column_metadata={
                "X": {"units": "mm"},
                "Y": {"units": "cm"},
                "Z": {"units": "m"},
            },
        )
        subset = matrix.subset_columns([0, 2])
        assert subset.n_variables == 2
        assert subset.column_metadata["X"]["units"] == "mm"
        assert subset.column_metadata["Z"]["units"] == "m"
        assert "Y" not in subset.column_metadata

    def test_impute_preserves_metadata(self):
        """测试impute方法保留metadata"""
        import numpy as np

        data = np.array([[1.0, np.nan], [3.0, 4.0]])
        matrix = DataMatrix(
            data,
            row_labels=["A", "B"],
            col_labels=["X", "Y"],
            metadata={"project": "Test"},
            specimen_metadata=[{"id": "S1"}, {"id": "S2"}],
            column_metadata={"X": {"units": "mm"}, "Y": {"units": "cm"}},
        )
        imputed = matrix.impute_mean()
        assert imputed.metadata == {"project": "Test"}
        assert imputed.specimen_metadata[0]["id"] == "S1"
        assert imputed.column_metadata["X"]["units"] == "mm"

    def test_transpose_swaps_metadata(self):
        """测试transpose交换specimen和column metadata"""
        spec_meta = [{"spec_id": "S1"}, {"spec_id": "S2"}]
        col_meta = {"X": {"col_units": "mm"}, "Y": {"col_units": "cm"}}

        matrix = DataMatrix(
            [[1.0, 2.0], [3.0, 4.0]],
            row_labels=["A", "B"],
            col_labels=["X", "Y"],
            specimen_metadata=spec_meta,
            column_metadata=col_meta,
        )
        transposed = matrix.transpose()

        # After transpose, original column metadata becomes specimen metadata
        assert transposed.specimen_metadata[0]["col_units"] == "mm"
        # And original specimen metadata becomes column metadata
        assert transposed.column_metadata["A"]["spec_id"] == "S1"


class TestDataMatrixMetadataEdgeCases:
    """测试边缘情况"""

    def test_empty_metadata(self):
        """测试空metadata"""
        matrix = DataMatrix([[1.0]])
        matrix.set_specimen_metadata(0, "key", "value")
        assert matrix.get_specimen_metadata(0)["key"] == "value"

    def test_metadata_setter(self):
        """测试metadata属性的setter"""
        matrix = DataMatrix([[1.0]])
        matrix.metadata = {"new": "value"}
        assert matrix.metadata == {"new": "value"}

    def test_specimen_metadata_setter(self):
        """测试specimen_metadata属性的setter"""
        matrix = DataMatrix([[1.0], [2.0]])
        new_spec_meta = [{"id": "A"}, {"id": "B"}]
        matrix.specimen_metadata = new_spec_meta
        assert matrix.specimen_metadata == new_spec_meta

    def test_column_metadata_setter(self):
        """测试column_metadata属性的setter"""
        matrix = DataMatrix([[1.0, 2.0]], col_labels=["X", "Y"])
        new_col_meta = {"X": {"units": "mm"}, "Y": {"units": "cm"}}
        matrix.column_metadata = new_col_meta
        assert matrix.column_metadata == new_col_meta

    def test_specimen_metadata_length_mismatch(self):
        """测试specimen_metadata长度不匹配"""
        from utils.exceptions import MatrixDimensionError

        with pytest.raises(MatrixDimensionError):
            DataMatrix(
                [[1.0], [2.0]],
                specimen_metadata=[{"id": "S1"}],  # 只有1个，但有2个样本
            )

    def test_column_metadata_keys_mismatch(self):
        """测试column_metadata键不匹配"""
        from utils.exceptions import MatrixDimensionError

        # Init does not validate - it just ignores mismatched keys
        # (uses empty dict for missing keys and ignores extra keys)
        matrix = DataMatrix(
            [[1.0, 2.0]],
            col_labels=["X", "Y"],
            column_metadata={"X": {}, "Z": {}},  # Z doesn't exist
        )
        # Z is ignored, X is kept
        assert "X" in matrix.column_metadata
        assert "Z" not in matrix.column_metadata

        # The setter DOES validate
        with pytest.raises(MatrixDimensionError):
            matrix.column_metadata = {"A": {}, "B": {}, "C": {}}  # Extra key "C"
