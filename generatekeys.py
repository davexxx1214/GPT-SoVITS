import secrets

# 生成密钥
auth_keys = [secrets.token_hex(16) for _ in range(100)]  # 生成100个密钥

# 保存到文件
with open('auth_keys.txt', 'w') as f:
    for key in auth_keys:
        f.write(key + '\n')

print("密钥生成完成并已保存到auth_keys.txt文件中.")