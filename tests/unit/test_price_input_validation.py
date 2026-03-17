from tools.df_price import DFPriceTools


def test_sanitize_object_name_accepts_normal_name():
    assert DFPriceTools._sanitize_object_name("QBZ95-1突击步枪") == "QBZ95-1突击步枪"
    assert DFPriceTools._sanitize_object_name("野蜂冲锋枪快拔套(绿)") == "野蜂冲锋枪快拔套(绿)"


def test_sanitize_object_name_rejects_illegal_chars():
    assert DFPriceTools._sanitize_object_name("非洲之心;DROP TABLE") == ""
    assert DFPriceTools._sanitize_object_name("\n\t") == ""


def test_sanitize_common_params_rejects_bad_id():
    safe, error = DFPriceTools._sanitize_common_params({"id": "abc-123"})
    assert safe == {}
    assert "id/objectId" in error


def test_sanitize_common_params_accepts_history_params():
    safe, error = DFPriceTools._sanitize_common_params(
        {
            "objectName": "非洲之心",
            "startTime": "2026-02-24",
            "endTime": "2026-03-03",
        }
    )
    assert error == ""
    assert safe["objectName"] == "非洲之心"
    assert safe["startTime"] == "2026-02-24"
    assert safe["endTime"] == "2026-03-03"
