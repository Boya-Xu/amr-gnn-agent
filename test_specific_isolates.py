# 解决导入路径问题
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.predict import get_prediction

if __name__ == "__main__":
    print("=== 测试指定菌株ID预测功能 ===")

    # 从你的预测结果中选3个已知存在的菌株ID
    test_isolates = [
        "1352.10008",
        "1352.10011",
        "1352.10018"
    ]

    print(f"测试菌株ID: {test_isolates}")
    print(f"预期预测数量: {len(test_isolates)}")

    try:
        result = get_prediction(
            feature_path="./data/extracted_unitigs",
            antibiotic="vancomycin",
            isolate_ids=test_isolates
        )

        print(f"\n✅ 预测成功！")
        print(f"实际预测数量: {len(result['y_proba'])}")
        print(f"返回的菌株ID: {result['isolate_ids']}")

        # 验证结果
        print("\n=== 结果验证 ===")
        if len(result['y_proba']) == len(test_isolates):
            print("✅ 预测数量与预期一致")
        else:
            print(f"❌ 预测数量不一致！预期{len(test_isolates)}，实际{len(result['y_proba'])}")

        # 检查返回的ID是否与传入的一致
        returned_ids = set(result['isolate_ids'])
        expected_ids = set(test_isolates)
        if returned_ids == expected_ids:
            print("✅ 返回的菌株ID与传入的完全一致")
        else:
            missing = expected_ids - returned_ids
            extra = returned_ids - expected_ids
            if missing:
                print(f"❌ 缺少以下菌株ID: {missing}")
            if extra:
                print(f"❌ 多出以下菌株ID: {extra}")

        # 打印详细预测结果
        print("\n=== 详细预测结果 ===")
        for i in range(len(result['isolate_ids'])):
            print(f"菌株: {result['isolate_ids'][i]}, "
                  f"敏感概率: {result['y_proba'][i][0]:.4f}, "
                  f"耐药概率: {result['y_proba'][i][1]:.4f}, "
                  f"预测结果: {'耐药' if result['y_pred'][i] == 1 else '敏感'}")

    except Exception as e:
        print(f"\n❌ 预测失败: {e}")
        import traceback

        traceback.print_exc()