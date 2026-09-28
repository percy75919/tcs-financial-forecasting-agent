from app.tools.financial_data_extractor import _table_metrics


def test_html_ifrs_table_extracts_latest_quarter_values():
    html = """
    <table>
        <tr>
            <th></th>
            <th>Three-month period ended Mar 31, 2025</th>
            <th>Dec 31, 2025</th>
            <th>Mar 31, 2026</th>
        </tr>
        <tr>
            <td>Revenue</td>
            <td>7,465</td>
            <td>7,509</td>
            <td>7,621</td>
        </tr>
        <tr>
            <td>Cost of revenue</td>
            <td>4,570</td>
            <td>4,444</td>
            <td>4,517</td>
        </tr>
        <tr>
            <td>Operating income</td>
            <td>1,807</td>
            <td>1,889</td>
            <td>1,927</td>
        </tr>
        <tr>
            <td>Net income</td>
            <td>1,418</td>
            <td>1,503</td>
            <td>1,479</td>
        </tr>
    </table>
    """

    revenue, net_income = _table_metrics(html)

    assert revenue == 7621
    assert net_income == 1479