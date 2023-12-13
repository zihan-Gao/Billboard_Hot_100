#获得网页数据
#获得user-agent
def get_user_agent():
    user_agent = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.93 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.93 Safari/537.36",
        "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:95.0) Gecko/20100101 Firefox/95.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:95.0) Gecko/20100101 Firefox/95.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Safari/605.1.15",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; Trident/7.0; rv:11.0) like Gecko",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.93 Safari/537.36 OPR/87.0.3819.143",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.93 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.93 Safari/537.36",
        "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:95.0) Gecko/20100101 Firefox/95.0"
    ]
    return random.choice(user_agent)





#获取网页内容
def get_request(url_text, user_agent):
    header = {"user-agent": user_agent}
    url = "https://www.billboard.com/charts/hot-100" + url_text
    try:
        request_result = requests.get(url, headers=header, timeout=180)
        if request_result.status_code != 200:
            print(request_result.status_code)
    except requests.exceptions.ConnectionError as e:
        print("ConnectionError:", e)
    except requests.exceptions.Timeout as e:
        print("Timeout:", e)
    except requests.exceptions.RequestException as e:
        print("RequestException:", e)
    request_result.encoding="utf-8"
    request_text = request_result.text
    return request_text

#获取歌曲歌名
def get_music_title(soup_text):
    music_title = []
    for i in soup_text:
        titles_text = i.find("li", class_="lrv-u-width-100p")
        music_title.append(titles_text.ul.li.h3.text.strip())
    return music_title

#获取歌曲歌手
def get_music_producer(soup_text):
    music_producer = []
    for i in soup_text:
        producer_text = i.find("li", class_="lrv-u-width-100p")
        music_producer.append(producer_text.ul.li.span.text.strip())
    return music_producer

#获取歌曲本周排名
def get_this_week_rank(soup_text):
    this_week_rank = []
    for i in soup_text:
        this_week_rank.append(i.ul.li.span.text.strip())
    return this_week_rank

#获取歌曲较上周排名变化
def get_this_week_to_last_week(soup_text):
    this_week_to_last_week = []
    for i in soup_text:
        if (i.ul.li.find("div")):
            this_week_to_last_week.append(i.li.div.svg.path["fill"])
        else:
            this_week_to_last_week.append(i.ul.li.find_all("span")[1].text.strip().replace("\n", ""))
    return this_week_to_last_week

#获取歌曲在本周的获奖情况
def get_this_week_award(soup_text):
    this_week_award = []
    for i in soup_text:
        award_text = i.find("li", class_="o-chart-results-list__item // a-chart-bg-color a-chart-color u-width-72 u-width-55@mobile-max u-width-55@tablet-only lrv-u-flex lrv-u-flex-shrink-0 lrv-u-align-items-center lrv-u-justify-content-center lrv-u-background-color-grey-lightest lrv-u-border-b-1 u-border-b-0@mobile-max lrv-u-border-color-grey-light lrv-u-flex-grow-1")
        if (award_text.find()):
            if (award_text.find("g")):
                this_week_award.append("other")
            else:
                this_week_award.append("star")
        else:
            this_week_award.append(None)
    return this_week_award

#获取歌曲在上周的排名
def get_last_week_rank(soup_text):
    last_week_rank = []
    for i in soup_text:
        summary_text = i.find("li", class_="lrv-u-width-100p u-hidden@tablet").find_all("li")
        last_week_rank.append(summary_text[2].span.text.strip())
    return last_week_rank

#获取歌曲的最佳排名
def get_best_rank(soup_text):
    best_rank = []
    for i in soup_text:
        summary_text = i.find("li", class_="lrv-u-width-100p u-hidden@tablet").find_all("li")
        best_rank.append(summary_text[3].span.text.strip())
    return best_rank
#获取歌曲在前100名的次数
def get_weeks_on_chart(soup_text):
    weeks_on_chart = []
    for i in soup_text:
        summary_text = i.find("li", class_="lrv-u-width-100p u-hidden@tablet").find_all("li")
        weeks_on_chart.append(summary_text[4].span.text.strip())
    return weeks_on_chart

#获得文件
def get_file(summary_data_list, url_text):
    summary_data_df = pandas.DataFrame([x for x in summary_data_list], columns=["music_title", "music_producer", "this_week_rank", "this_week_to_last_week", "this_week_award", "last_week_rank", "best_rank", "weeks_on_chart"])
    csv_file = "summary_data/" + "summary_data_" + url_text.strip("/") + ".csv"
    xlsx_file = "summary_data/" + "summary_data" + url_text.strip("/") + ".xlsx"
    summary_data_df.to_csv(csv_file, index=False)
    summary_data_df.to_excel(xlsx_file, index=False)

#获得日期列表
def get_dates_list():
    dates_list = []
    year = int(input())
    for month in range(1, 13):
        for i in calendar.monthcalendar(year, month):
            #这天的星期六存在
            if i[calendar.SATURDAY] != 0:
                dates_list.append(f"/{year}-{month:02d}-{i[calendar.SATURDAY]:02d}/")
    return dates_list





#main程序
def main():
    dates_list = get_dates_list()
    for url_text in dates_list:
        summary_data_list = []
        user_agent = get_user_agent()
        request_text = get_request(url_text, user_agent)
    
        request_soup = BeautifulSoup(request_text, "html.parser")
        soup_text = request_soup.find_all("div", class_="o-chart-results-list-row-container")
    
        music_title = get_music_title(soup_text)
        music_producer = get_music_producer(soup_text)
        this_week_rank = get_this_week_rank(soup_text)
        this_week_to_last_week = get_this_week_to_last_week(soup_text)
        this_week_award = get_this_week_award(soup_text)
        last_week_rank = get_last_week_rank(soup_text)
        best_rank = get_best_rank(soup_text)
        weeks_on_chart = get_weeks_on_chart(soup_text)
    
        summary_data_list = list(zip(music_title, music_producer, this_week_rank, this_week_to_last_week, this_week_award, last_week_rank, best_rank, weeks_on_chart))
        get_file(summary_data_list, url_text)
        time.sleep(60)
