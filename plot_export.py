import pandas as pd
import matplotlib.pyplot as plt

# ==============================================================================
# 1. 엑셀 파일 불러오기 (여기에 본인 파일 경로를 적으세요!)
# ==============================================================================
# 예: "C:/Users/내이름/Desktop/실험결과_빨강.xlsx"
# 파일 경로에 한글이 있어도 잘 읽힙니다.

try:
    # 빨간색 데이터 (최적화 모델)
    df_red = pd.read_csv("dt0.1rollout2048.csv") 
    
    # 파란색 데이터 (기본 모델)
    df_blue = pd.read_csv("baseline.csv")
    
    print("✅ 엑셀 파일 읽기 성공!")
    print(f"빨강 데이터 개수: {len(df_red)}개")
    print(f"파랑 데이터 개수: {len(df_blue)}개")

except FileNotFoundError:
    print("❌ 파일을 찾을 수 없습니다. 파일 경로와 이름을 다시 확인해주세요.")
    exit()

# ==============================================================================
# 2. 이동 평균 (Smoothing) 적용
# ==============================================================================
# 데이터가 촘촘하면 숫자를 키우세요 (예: 10 ~ 50)
# 데이터가 별로 없으면 줄이세요 (예: 3 ~ 5)
WINDOW_SIZE = 5 

# 'Success'는 엑셀 파일의 B열 제목(Header)이어야 합니다. 다르면 수정하세요!
# 예: 엑셀 제목이 'Value'라면 -> df_red['Value']
col_name_step = 'Step'      # 엑셀의 스텝 열 이름
col_name_score = 'Success'  # 엑셀의 성공률 열 이름

try:
    df_red['Smoothed'] = df_red[col_name_score].rolling(window=WINDOW_SIZE, min_periods=1).mean()
    df_blue['Smoothed'] = df_blue[col_name_score].rolling(window=WINDOW_SIZE, min_periods=1).mean()
except KeyError:
    print(f"❌ 엑셀 파일의 열 이름이 '{col_name_score}'가 맞나요? 확인해주세요.")
    exit()

# ==============================================================================
# 3. 그래프 그리기
# ==============================================================================
plt.figure(figsize=(10, 6))
plt.grid(True, linestyle='--', alpha=0.5)

# 빨강 (Optimized)
plt.plot(df_red[col_name_step], df_red['Smoothed'], 
         color='red', label='Optimized (dt=0.1, Roll=2048)', linewidth=2)
# 원본 데이터 점 (투명하게)
plt.scatter(df_red[col_name_step], df_red[col_name_score], color='red', alpha=0.1, s=10)

# 파랑 (Baseline)
plt.plot(df_blue[col_name_step], df_blue['Smoothed'], 
         color='blue', label='Baseline (dt=0.01, Roll=4096)', linewidth=2)
# 원본 데이터 점 (투명하게)
plt.scatter(df_blue[col_name_step], df_blue[col_name_score], color='blue', alpha=0.1, s=10)

plt.title("Comparison of Learning Efficiency", fontsize=14)
plt.xlabel("Total Environmental Steps", fontsize=12)
plt.ylabel("Success Rate (Moving Average)", fontsize=12)
plt.legend(fontsize=12)

# 저장 및 보여주기
plt.savefig("result_graph.png", dpi=300)
print("✅ 그래프가 'result_graph.png'로 저장되었습니다.")
plt.show()