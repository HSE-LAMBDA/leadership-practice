import indicoio
indicoio.config.api_key = '7f666c40534f334fd0c8309c18b24ba1'
print(
indicoio.sentiment(u"я люблю тебя!", language='ru')
)
