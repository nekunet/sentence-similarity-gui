import threading

import gensim
import MeCab
import PySimpleGUI as sg


def set_dpi_awareness():
    """
    高DPIに対応する関数
    """
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass


def split_into_words(text):
    """
    configから入力ファイル群を読み込む関数

    Args:
        text: 入力文

    Returns:
        words: 名詞のみ抽出した単語のリスト
    """
    words = []
    tagger = MeCab.Tagger('mecabrc')
    tagger.parse('')
    node = tagger.parseToNode(text)

    while node:
        #print(node.surface, node.feature)
        meta = node.feature.split(",")[0]
        if meta == "名詞":
            words.append(node.surface)
        node = node.next

    return words


def read_file(filename):
    """
    テキストファイルの内容を読む関数

    Args:
        filename: ファイル名

    Returns:
        lines: 読み込んだテキストが1行ごとに格納されているリスト
    """
    lines = []

    with open(filename, "r", encoding="utf-8") as f:
        for line in f.readlines():
            line = line.rstrip()
            lines.append(line)

    return lines


def make_word_pairs(words1, words2):
    """
    ワードのペアを作る関数

    Args:
        words1: 単語のリスト1
        words2: 単語のリスト2 
    
    Returns:
        word_pairs: 単語のペアリスト
    """
    word_pairs = []

    for word1 in words1:
        for word2 in words2:
            word_pairs.append([word1, word2])

    return word_pairs


def calc_similarity_score(model_file, word_pairs):
    """
    単語のペアから類似度を計算する関数

    Args:
        model_file: word2vecのmodelファイル（バイナリ）
        word_pairs: 単語のペアリスト

    Returns:
        similarity_score: 類似度スコア
    """
    # modelをバイナリモードで読み込む
    model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=True)

    # モデルに含まれている単語
    vocab_list = list(model.index_to_key)

    cnt = 0
    similarity_score = 0 

    for word1, word2 in word_pairs:
        # 2つの単語がモデルに含まれている場合のみ処理
        if (word1 in vocab_list) and (word2 in vocab_list):
            similarity_score += model.similarity(word1, word2)
            cnt += 1
    
    if cnt != 0:
        similarity_score /= cnt

    return similarity_score


def write_file_and_window(filename, result, threshold, window):
    """
    結果を出力する関数

    Args:
        filename: 出力するファイル名
        result: [類似度, 文1, 文2]のリスト
        threshold: しきい値 
        window: pysimpleguiのウィンドウ
    """
    threshold_over_flg = False
    window['-LOG-'].update("")
    
    with open(filename, "w", encoding="utf-8") as f:
        for similarity_score, line1, line2 in result:
            f.write("{0},{1},{2}\n".format(similarity_score, line1, line2))
            window['-LOG-'].print("{0},{1},{2}\n".format(similarity_score, line1, line2))
            
            # しきい値チェック
            if similarity_score > threshold:
                threshold_over_flg = True
        
        if threshold_over_flg:
            window['-LOG-'].print("OK")
            f.write("OK\n")


def run(lock, window, input_txt1, input_txt2, model_file, threshold, output_txt):
    """
    引数を受け取り、処理を実行する関数

    Args:
        lock: threadのlock
        window: pysimpleguiのwindow
        input_txt1: 入力ファイル1のファイルパス
        input_txt2: 入力ファイル2のファイルパス
        model_file: モデルファイルのファイルパス
        threshold: しきい値
        output_txt: 出力ファイルのファイルパス
    """

    # 既にアプリケーションが起動してた場合は実行しない（バッチ的な処理をしない）
    if lock.locked():
        window.write_event_value('-THREAD-', "ERROR")
        return

    # threadをlock
    lock.acquire()

    # 処理実行
    lines1 = read_file(input_txt1)
    lines2 = read_file(input_txt2)
    result = []

    # プログレスバーの初期化
    num = len(lines1)*len(lines2)
    cnt = 0
    window['-PROGBAR-'].update_bar(cnt, max=num)

    for line1 in lines1:
        for line2 in lines2:
            words1 = split_into_words(line1)
            words2 = split_into_words(line2)

            word_pairs = make_word_pairs(words1, words2)
            
            similarity_score = calc_similarity_score(model_file, word_pairs)

            # 結果をリストに格納
            result.append([similarity_score, line1, line2])

            # プログレスバーを進捗させる
            cnt += 1
            window['-PROGBAR-'].update_bar(cnt, max=num)
    
    # 類似度でソート
    result.sort(reverse=True, key=lambda x: x[0])

    # 最大の類似度を代入
    global max_similality_score
    max_similality_score = result[0][0]

    # 結果をファイルとウィンドウに表示
    write_file_and_window(output_txt, result, threshold, window)
    
    # 正常終了
    window.write_event_value('-THREAD-', "OK")

    # lockの解放
    lock.release()

    return


if __name__ == "__main__":
    """
    メイン関数
    """
    # 高DPIに対応
    set_dpi_awareness()

    # threadの競合を回避するためのlock
    lock = threading.Lock()
    
    # デザインテーマの設定
    sg.theme("SystemDefault1")

    # 最大の類似度
    max_similality_score = 0

    # ウィンドウレイアウト
    layout = [
              [sg.Text('入力ファイル1', size=(11, 0),), 
               sg.InputText(key='-INPUT_FILE1-'), 
               sg.FileBrowse('Browse', key='-FILE_BROWSE1-', initial_folder="./", 
                             file_types=(("Text Files", "*.txt"),))],

              [sg.Text('入力ファイル2', size=(11, 0)), 
               sg.InputText(key='-INPUT_FILE2-'), 
               sg.FileBrowse('Browse', key='-FILE_BROWSE2-', initial_folder="./", 
                             file_types=(("Text Files", "*.txt"),))],

              [sg.Text('モデルファイル', size=(11, 0), pad=(5, 6)), 
               sg.InputText('model.bin', key='-MODEL_FILE-', pad=(5, 6))],

              [sg.Text('しきい値', size=(11, 0), pad=(5, 14)), 
               sg.InputText('0.2', key='-THRESHOLD-', size=(6, 0), pad=(5, 14))],

              [sg.Text('出力ファイル (*.csv)', size=(16, 0)), 
               sg.InputText('result.csv', key='-OUTPUT_FILE-', size=(40, 0))],

              [sg.Multiline('', key='-LOG-', autoscroll=True, size=(80, 12), pad=(5, 23))],

              [sg.Column([[sg.ProgressBar(999, key="-PROGBAR-", orientation='h', size=(35, 10))]],
                          vertical_alignment='center', justification='center', pad=((0, 0), (0, 20)))],

              [sg.Column([[sg.Button('実行', size=(7, 1), pad=(12, 0)),
                           sg.Button('確認', size=(7, 1), pad=(12, 0))]],
                         vertical_alignment='center', justification='center', pad=(0, 10))]
             ]

    # ウィンドウの生成
    window = sg.Window("GUI tool", layout, finalize=True, font=("Meiryo UI", 9), margins=(20, 20))

    # イベントループ
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED:
            break
        
        # 実行ボタン押下時の処理
        elif event == '実行':
            # GUIからの設定値読み込みと整形
            input_txt1 = values['-INPUT_FILE1-']
            input_txt2 = values['-INPUT_FILE2-']
            model_file = values['-MODEL_FILE-']
            threshold = float(values['-THRESHOLD-'])
            output_txt = values['-OUTPUT_FILE-']

            # スレッドの実行
            t = threading.Thread(target=run, args=(lock, window, input_txt1, input_txt2,
              model_file, threshold, output_txt), daemon=True)
            t.start()
        
        # 確認ボタン押下時の処理
        elif event == '確認':
            if max_similality_score > float(values['-THRESHOLD-']):
                sg.Window("Info", [[sg.T('しきい値チェック判定：OK')], [sg.Button("OK")]], 
                    disable_close=False).read(close=True)
            else:
                sg.Window("Info", [[sg.T('しきい値チェック判定：NG')], [sg.Button("OK")]], 
                    disable_close=False).read(close=True)
        
        # threadから返答が来た時
        if event == '-THREAD-':
            if values['-THREAD-'] == "OK":
                # 完了メッセージ
                sg.Window("Info", [[sg.T('実行処理が終了しました。　　　')], [sg.Button("OK")]], 
                    disable_close=False).read(close=True)
            # threadのlockを取得できなかった場合
            else:
                # エラーメッセージ
                sg.Window("Error", [[sg.T('実行中の処理があります。　　　')], [sg.Button("OK")]], 
                    disable_close=False).read(close=True)
    
    window.close()
