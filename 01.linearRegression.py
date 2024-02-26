# 필요한 라이브러리 임포트
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 임의의 데이터 생성
X = [[29],[26],[34],[31],[25],[29],[32],[31],[24],[33],[25],[31],[26],[30]]
y = [[77],[62],[93],[84],[59],[64],[80],[76],[58],[91],[51],[73],[65],[84]]
# 데이터 시각화
plt.scatter(X, y)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Generated Data')
plt.show()

# 데이터를 훈련 세트와 테스트 세트로 나누기
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 선형 회귀 모델 만들기 및 훈련
model = LinearRegression()
model.fit(X_train, y_train)

#훈련된 회귀식 정리
print("절편",model.intercept_)  #편향(절편)
print("기울기",model.coef_)     #기울기
print("R값",model.score(X, y))  # R값 점수

plt.plot(X_train, y_train, 'o')
plt.plot(X_train,model.predict(X_train))
plt.title('Linear Regression Trained') # 학습된 회귀선
plt.show()

# 훈련된 모델로 예측
y_pred = model.predict(X_test)

# 테스트 세트의 성능 평가
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error on Test Set: {mse}')

# 훈련된 선형 회귀 선 시각화
plt.scatter(X_test, y_test)
plt.plot(X_test, y_pred, color='red', linewidth=3)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression Prediction')
plt.show()