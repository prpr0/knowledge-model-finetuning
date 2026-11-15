"""官方语料与分区规则的临时合成数据测试。"""
import hashlib
import unittest

from knowledge_model_finetuning.corpus import assign_splits, validate_document


def document(index, text=None):
    """构造仅用于规则测试的合成文档。"""
    text = text or f"独立资料{index}：" + "".join(chr(0x4e00 + index * 100 + offset) for offset in range(80))
    return dict(id=f"d{index}", source_url=f"https://docs.python.org/zh-cn/3.11/library/test{index}.html", fetched_at="2026-11-15T10:00:00+00:00", version="3.11", section=f"章节{index}", license="PSF-2.0", attribution="Python Software Foundation", transformation="仅提取正文", text=text, sha256=hashlib.sha256(text.encode()).hexdigest())


class CorpusTests(unittest.TestCase):
    """检查来源、哈希和先文档分组的行为。"""

    def test_source_and_hash(self):
        """拒绝伪域名及正文篡改。"""
        item = document(1)
        self.assertEqual(validate_document(item)["text"], item["text"])
        for changes in ({"source_url": "https://docs.python.org.evil/zh-cn/3.11/a"}, {"text": "篡改"}, {"version": "unknown"}):
            with self.assertRaises(ValueError):
                validate_document(item | changes)

    def test_stable_near_duplicate_groups(self):
        """近重复组不受输入顺序影响且不可跨区。"""
        items = [document(i) for i in range(12)]
        items += [document(20, items[0]["text"] + "补充")]
        first = assign_splits(items)
        second = assign_splits(list(reversed(items)))
        self.assertEqual({d["id"]: (d["group_id"], d["split"]) for d in first}, {d["id"]: (d["group_id"], d["split"]) for d in second})
        self.assertEqual(first[0]["group_id"], first[-1]["group_id"])
        self.assertEqual({d["split"] for d in first}, {"train", "validation", "test"})
        with self.assertRaises(ValueError):
            assign_splits(items[:2])
